// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "ckernel_sfpu_log1p.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_logsigmoid(
    const std::uint32_t dst_index_in0,  // Index for input (x)
    const std::uint32_t dst_index_in1,  // Index for exp(-|x|)
    const std::uint32_t dst_index_out)  // Index for output
{
    // Exact formulation: logsigmoid(x) = min(x, 0) - log1p(exp(-|x|))
    // Eliminates 0.91% fp32 relative error, removes x <= -4 drop, pins error within <= 2.0 ULP (#53787)
    constexpr std::uint32_t dst_tile_size_sfpi = 32;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat u = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        sfpi::vFloat min_x_0 = sfpi::min(x, 0.0f);
        sfpi::vFloat log1p_u = calculate_log1p_fp32<true>(u);

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = min_x_0 - log1p_u;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void logsigmoid_init() {}

}  // namespace sfpu
}  // namespace ckernel
