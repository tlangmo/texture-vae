/*
 * Copyright 2011 The Emscripten Authors.  All rights reserved.
 * Emscripten is available under two separate licenses, the MIT license and the
 * University of Illinois/NCSA Open Source License.  Both these licenses can be
 * found in the LICENSE file.
 */

#include <iostream>
#include <string>
#include <chrono>
#include <random>
#include <emscripten.h>

static std::default_random_engine generator;

extern "C"
int EMSCRIPTEN_KEEPALIVE vaeSeed()
{
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
    return 0;
}

extern "C"
double EMSCRIPTEN_KEEPALIVE vaeRandn(float mean, float stdDev)
{
    std::normal_distribution<double> distribution(mean,stdDev);
    double number = distribution(generator);
    return number;

  return 0;
}