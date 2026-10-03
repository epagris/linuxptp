/**
 * @file kalman.h
 * @note Copyright (C) 2025 András Wiesner <wabe@mit.bme.hu>
 * @note SPDX-License-Identifier: GPL-2.0+
 *
 */

#ifndef HAVE_KALMAN_H
#define HAVE_KALMAN_H

#include "servo.h"

struct servo *kalman_servo_create(struct config *cfg, double fadj);

#endif /* HAVE_KALMAN_H */
