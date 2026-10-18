# Expansion-utilisation harness (docs/expanded_k/STATUS.md); copy to tests/ and apply expk_util_probe.patch. One eager forward with the device profiler on, for the expansion / QK sum zones.
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
