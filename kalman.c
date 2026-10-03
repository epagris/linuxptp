/**
 * @file kalman.c
 * @brief Implements the PTP-tailored Kalman-filter designed and presented
 * by Giada Giorgi and Claudio Narduzzi in the paper Performance Analysis
 * of Kalman-Filter-Based Clock Synchronization in IEEE 1588 Networks. The
 * implementation follows the equation numbering and naming introduced in
 * the above article. Some linear algebra operation have been implemented
 * to simplify the evaluation of the matrix equations without linking
 * against a full-featured linear algebra library like CBLAS or LAPACK.
 * @note Copyright (C) 2025 András Wiesner <wabe@mit.bme.hu>
 * @note SPDX-License-Identifier: GPL-2.0+
 *
 */

#include "kalman.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"
#include "contain.h"
#include "print.h"
#include "servo.h"
#include "servo_private.h"

/* ---------------- */

/* predefined matrix sizes */
#define MTX_M (2)
#define MTX_N (2)

typedef double mtx[MTX_M][MTX_N];
typedef double vec[MTX_M];

static void mtx_copy(mtx dst, mtx src)
{
	unsigned int i, j;

	for (i = 0; i < MTX_M; i++) {
		for (j = 0; j < MTX_N; j++) {
			dst[i][j] = src[i][j];
		}
	}
}

static void mtx_add(mtx r, mtx a, mtx b)
{
	unsigned int i, j;

	for (i = 0; i < MTX_M; i++) {
		for (j = 0; j < MTX_N; j++) {
			r[i][j] = a[i][j] + b[i][j];
		}
	}
}

static void mtx_sub(mtx r, mtx a, mtx b)
{
	unsigned int i, j;

	for (i = 0; i < MTX_M; i++) {
		for (j = 0; j < MTX_N; j++) {
			r[i][j] = a[i][j] - b[i][j];
		}
	}
}

static void mtx_scale(mtx r, double c, mtx a)
{
	unsigned int i, j;

	for (i = 0; i < MTX_M; i++) {
		for (j = 0; j < MTX_N; j++) {
			r[i][j] = c * a[i][j];
		}
	}
}

static void mtx_dot(vec r, mtx m, vec v)
{
	unsigned int i;

	for (i = 0; i < MTX_M; i++) {
		r[i] = m[i][0] * v[0] + m[i][1] * v[1];
	}
}

static void mtx_mul(mtx r, mtx a, mtx b)
{
	r[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0];
	r[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1];
	r[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0];
	r[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1];
}

static void mtx_transp(mtx r, mtx m)
{
	unsigned int i, j;

	for (i = 0; i < MTX_M; i++) {
		for (j = 0; j < MTX_N; j++) {
			r[i][j] = m[j][i];
		}
	}
}

static double mtx_det(mtx m)
{
	return m[0][0] * m[1][1] - m[0][1] * m[1][0];
}

static void mtx_inverse(mtx r, mtx m)
{
	r[0][0] = m[1][1];
	r[0][1] = -m[0][1];
	r[1][0] = -m[1][0];
	r[1][1] = m[0][0];

	mtx_scale(r, 1 / mtx_det(m), r);
}

static void mtx_zero(mtx m)
{
	unsigned int i, j;

	for (i = 0; i < MTX_M; i++) {
		for (j = 0; j < MTX_N; j++) {
			m[i][j] = 0;
		}
	}
}

static void mtx_unit(mtx m)
{
	unsigned int i, j;

	for (i = 0; i < MTX_M; i++) {
		for (j = 0; j < MTX_N; j++) {
			m[i][j] = i == j;
		}
	}
}

static void vec_copy(vec r, vec v)
{
	unsigned int i;

	for (i = 0; i < MTX_M; i++) {
		r[i] = v[i];
	}
}

static void vec_add(vec r, vec a, vec b)
{
	unsigned int i;

	for (i = 0; i < MTX_M; i++) {
		r[i] = a[i] + b[i];
	}
}

static void vec_sub(vec r, vec a, vec b)
{
	unsigned int i;

	for (i = 0; i < MTX_M; i++) {
		r[i] = a[i] - b[i];
	}
}

static void vec_zero(vec v)
{
	unsigned int i;

	for (i = 0; i < MTX_M; i++) {
		v[i] = 0;
	}
}

/* -------------------- */

#define SQR(x) ((x) * (x))

/* -------------------- */

struct kalman_servo {
	struct servo servo;

	mtx A;      /* system matrix */
	mtx B;      /* input matrix */
	vec u;      /* control input */
	mtx Q;      /* process noise covariance */
	mtx K;      /* Kalman gain */
	mtx P;      /* a posteriori error covariance matrix */
	mtx P_pri;  /* a priori P error covariance matrix */
	mtx R;      /* measurement covariance matrix */
	mtx I;      /* identity matrix */
	vec x_pri;  /* a priori state estimator */
	vec x;      /* a posteriori state estimator */

	double sigma_theta_squared;    /* offset process variance (norm.) */
	double sigma_gamma_squared;    /* skew process variance (norm.) */
	double sigma_theta_m_squared;  /* measurement variance (norm.) */

	/* threshold for switching between fast and calm clock adjustment modes */
	uint64_t fast_tuning_threshold_ns;
	/* fast tuning coefficient for incorporating the offset */
	double fast_tuning_offset_coefficient;
	/* calm tuning coefficient for incorporating the offset */
	double calm_tuning_offset_coefficient;

	uint64_t cycle;        /* cycle counter */
	int64_t offset[2];     /* current and previous offsets */
	uint64_t local_ts[2];  /* current and previous local timestamps */
	double sync_interval;  /* synchronization interval */
	double freq_ppb;       /* frequency ppb */
};

static void kalman_destroy(struct servo *servo)
{
	/* free the resources */
	struct kalman_servo *s = container_of(servo, struct kalman_servo, servo);

	free(s);
}

static void kalman_reset_state(struct kalman_servo *s)
{
	/* reset matrices and vectors */
	mtx_unit(s->A);
	mtx_unit(s->B);
	vec_zero(s->u);
	mtx_zero(s->Q);
	mtx_zero(s->K);
	mtx_zero(s->P);
	mtx_zero(s->P_pri);
	mtx_zero(s->R);
	mtx_unit(s->I);
	vec_zero(s->x_pri);
	vec_zero(s->x);

	/* reset cycle count and time offset array */
	s->cycle = 0;

	memset(s->offset, 0, sizeof(s->offset));
	memset(s->local_ts, 0, sizeof(s->local_ts));
}

static void kalman_adjust_sync_interval(struct kalman_servo *s)
{
	s->A[0][1] = s->sync_interval;
	s->B[0][1] = -s->sync_interval;

	s->Q[0][0] = s->sync_interval * s->sigma_theta_squared;
	s->Q[1][1] = s->sync_interval * s->sigma_gamma_squared;

	s->R[0][0] = s->sigma_theta_m_squared;
	s->R[0][1] = s->R[1][0] = s->sigma_theta_m_squared / s->sync_interval;
	s->R[1][1] = 2.0 * s->sigma_theta_m_squared / SQR(s->sync_interval);
}

static double kalman_sample(struct servo *servo,
			    int64_t offset,
			    uint64_t local_ts,
			    double weight,
			    enum servo_state *state)
{

	struct kalman_servo *s = container_of(servo, struct kalman_servo, servo);

	double tuning_ppb = 0.0;	/* result variables */
	double tuning = 0.0;
	uint64_t sync_interval_ns;	/* sync interval */
	double skew;
	double z[2];			/* measurement vector */
	double tuning_coefficient;
	double relative_time_error; /* time error versus sync interval ratio */
	double calculated_freq_ppb; /* calculated frequency adjustment */

	/* vectors and matrices, intermediate values */
	vec Ax, Bu;
	mtx AP, At, APAt;
	mtx P_priR, P_priRinv;
	vec zxpri, Kzxpri;
	mtx IK;

	/* ---- */

	/* retain offset and local timestamp */
	s->offset[0] = offset;
	s->local_ts[0] = local_ts;

	/* check if the current sample is newer than the previous one */
	if (s->local_ts[0] <= s->local_ts[1]) {
		/* if not, then go into UNLOCKED state */
		*state = SERVO_UNLOCKED;

		/* do not retain faulty sync cycle data */
		goto retain_cycle_data;
	}

	/* jump the clock if error is too high */
	if ((servo->first_update && servo->first_step_threshold &&
		servo->first_step_threshold < llabs(offset)) ||
		(servo->step_threshold && servo->step_threshold < llabs(offset))) {
		*state = SERVO_JUMP;

		/* reset the servo */
		kalman_reset_state(s);

		/* return with the last clock adjustment */
		return s->freq_ppb;
	}

	/* in the very first cycle skip running the filter */
	if (s->cycle == 0) {
		*state = SERVO_UNLOCKED;
		goto retain_cycle_data;
	}

	/* ---- PREPARE THE PARAMETERS ---- */

	/* calculate input data */
	sync_interval_ns = s->local_ts[0] - s->local_ts[1];

	s->sync_interval = sync_interval_ns * 1E-09;
	skew = ((double) (s->offset[0] - s->offset[1])) /
		((double) (sync_interval_ns));  /* skew */

	/* compose the measurement vector */
	z[0] = s->offset[0] * 1E-09;  /* offset in seconds */
	z[1] = skew;                  /* skew */

	/* inject sync interval into the relevant places */
	kalman_adjust_sync_interval(s);

	/* incorporate weight into the calculations */
	mtx_scale(s->R, 1 / weight, s->R);

	/* ---- RUN THE FILTER ---- */

	if (s->cycle == 1) {
		mtx_copy(s->P, s->Q);  /* P(1|0) = Q */
		vec_copy(s->x, z);     /* x = z */
	}

	/* prediction equations */

	/* (27) */
	mtx_dot(Ax, s->A, s->x);    /* Ax = A * x(n-1) */
	mtx_dot(Bu, s->B, s->u);    /* Bu = B * u(n-1) */
	vec_add(s->x_pri, Ax, Bu);  /* x(n|n-1) = Ax + Bu */

	/* (28) */
	mtx_mul(AP, s->A, s->P);        /* AP = A * P */
	mtx_transp(At, s->A);           /* At = A' */
	mtx_mul(APAt, AP, At);          /* APAt = AP * At */
	mtx_add(s->P_pri, APAt, s->Q);  /* P(n|n-1) = APAt + Q */

	/* correction equations */

	/* (30) */
	mtx_add(P_priR, s->P_pri, s->R);     /* P_priR = P(n|n-1) + R */

	/* check if P_priR is positive definite */
	if ((mtx_det(P_priR) > 0) && (P_priR[0][0] > 0)) {
		mtx_inverse(P_priRinv, P_priR);  /* P_priRinv = (P_priR)^(-1) */
	} else { /* if P_priR is not positive definite */
		/* report UNLOCKED state */
		*state = SERVO_UNLOCKED;

		/* log the Kalman filter error */
		pr_warning("Kalman filter P_priR is not positive, skipping cycle");

		/* It is safe to abandon the Kalman-filter algorithm
		 * at this specific point and just retain the newly received
		 * sync cycle data. No clock adjustment will be issued. */
		goto retain_cycle_data;
	}

	mtx_mul(s->K, s->P_pri, P_priRinv);  /* K = P(n|n-1) * P_priRinv */

	/* (31) */
	vec_sub(zxpri, z, s->x_pri);      /* zxpri = z - x(n|n-1) */
	mtx_dot(Kzxpri, s->K, zxpri);     /* Kzxpri = K * zxpri */
	vec_add(s->x, s->x_pri, Kzxpri);  /* x = x(n|n-1) + Kzxpri */

	/* (32) */
	mtx_sub(IK, s->I, s->K);      /* IK = I - K */
	mtx_mul(s->P, IK, s->P_pri);  /* P = IK * P(n|n-1) */

	/* ---- TUNING ---- */

	/* determine tuning coefficient */
	tuning_coefficient =
		((fabs(s->x[0]) * 1E+09) > s->fast_tuning_threshold_ns)
		? s->fast_tuning_offset_coefficient
		: s->calm_tuning_offset_coefficient;

	/* calculate relative time error, (change x[0] into nanoseconds) */
	relative_time_error = (-s->x[0] * 1E+09) / ((double) sync_interval_ns);

	/* calculate final, unitless tuning */
	tuning = -s->x[1] + relative_time_error * tuning_coefficient;

	/* scale clock tuning into ppbs */
	tuning_ppb = tuning * 1E+09;

	/* feed back tuning */
	s->u[0] = 0;
	s->u[1] = tuning;

	/* the servo is currently driving the clock */
	*state = SERVO_LOCKED;

	/* ---- DATA RETENTION ---- */

retain_cycle_data:
	s->offset[1] = s->offset[0];
	s->local_ts[1] = s->local_ts[0];
	s->cycle++;

	/* accumulate and limit clock adjustment */
	calculated_freq_ppb = s->freq_ppb - tuning_ppb;

	/* lower limit */
	if (calculated_freq_ppb < -servo->max_frequency) {
		s->freq_ppb = -servo->max_frequency;
	/* upper limit */
	} else if (calculated_freq_ppb > servo->max_frequency) {
		s->freq_ppb = servo->max_frequency;
	/* normal operation */
	} else {
		s->freq_ppb = calculated_freq_ppb;
	}

	return s->freq_ppb;
}

static void kalman_reset(struct servo *servo)
{
	struct kalman_servo *s =
		container_of(servo, struct kalman_servo, servo);

	kalman_reset_state(s);
}

static void kalman_sync_interval(struct servo *servo, double interval)
{
	(void) servo;
	(void) interval;

	/* No computation or adjustment is done here, since the actual,
	 * sample-based sync intervals are determined in the kalman_sample()
	 * call each sync cycle. */
}

static void kalman_init_variances(struct config *cfg, struct kalman_servo *s)
{
	s->sigma_theta_squared =
		config_get_double(cfg, NULL, "kalman_sigma_theta_squared");
	s->sigma_gamma_squared =
		config_get_double(cfg, NULL, "kalman_sigma_gamma_squared");
	s->sigma_theta_m_squared =
		config_get_double(cfg, NULL, "kalman_sigma_theta_m_squared");
}

static void kalman_init_fast_tuning_params(struct config *cfg,
	struct kalman_servo *s)
{
	s->fast_tuning_threshold_ns =
		config_get_uint(cfg, NULL, "kalman_fast_tuning_threshold");
	s->fast_tuning_offset_coefficient =
		config_get_double(cfg, NULL, "kalman_fast_tuning_offset_coeff");
	s->calm_tuning_offset_coefficient =
		config_get_double(cfg, NULL, "kalman_calm_tuning_offset_coeff");
}

struct servo *kalman_servo_create(struct config *cfg, double fadj)
{
	struct kalman_servo *s;

	s = calloc(1, sizeof(struct kalman_servo));
	if (!s) {
		return NULL;
	}

	s->servo.destroy = kalman_destroy;
	s->servo.sample = kalman_sample;
	s->servo.sync_interval = kalman_sync_interval;
	s->servo.reset = kalman_reset;
	s->freq_ppb = fadj;

	kalman_init_variances(cfg, s);
	kalman_init_fast_tuning_params(cfg, s);
	kalman_reset(&s->servo);

	return &s->servo;
}
