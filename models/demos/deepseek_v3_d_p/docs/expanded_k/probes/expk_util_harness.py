# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Expansion-utilisation harness (docs/expanded_k/STATUS.md): one eager forward with the device profiler on, for the
# expansion / QK sum zones. It needs the model's test fixtures, so from the repo root:
#   git apply models/demos/deepseek_v3_d_p/docs/expanded_k/probes/expk_util_probe.patch
#   cp models/demos/deepseek_v3_d_p/docs/expanded_k/probes/expk_util_harness.py \
#      models/demos/deepseek_v3_d_p/tests/scratch_expk_util.py
#   TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_SUM=1 scripts/run_safe_pytest.sh \
#      models/demos/deepseek_v3_d_p/tests/scratch_expk_util.py -k k2_7   # then -k k3, one case per run
# Each run writes generated/profiler/.logs/profile_log_device.csv; analyse it before the next run overwrites it.
import pytest
from loguru import logger

import models.demos.deepseek_v3_d_p.tests.test_mla as T
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [pytest.param((8, 4), torus_xy_device_params(l1_small_size=1152), id="torus-xy-8x4")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "variant,prefill_len",
    [pytest.param("kimi_k3", 512 * 1024, id="k3-512k"), pytest.param("kimi_k2_7", 256 * 1024, id="k2_7-256k")],
    indirect=["variant"],
)
@pytest.mark.timeout(0)
def test_expk_util(request, mesh_device, device_params, variant, prefill_len, monkeypatch):
    monkeypatch.setenv("PREFILL_MLA_EXPANDED_K", "1")
    T._run_chunked_prefill(
        request,
        mesh_device,
        reference=None,
        topology=per_axis_topology(device_params["fabric_config"]),
        iters_isl=[5120],
        prefill_len=prefill_len,
        use_metadata_tensor=True,
        tp_shard_kv=False,
        num_cache_layers=1,
    )
    logger.info(f"RESULT done {variant.name} {prefill_len}")
