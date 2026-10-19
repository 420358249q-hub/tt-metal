# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Host only: the signposted-region filter of run_model_device_perf_test_with_merge.
import pandas as pd
import pytest

from models.demos.deepseek_v3_d_p.utils.perf_utils import ops_in_signposted_region

_REGION = ("MLA_START", "MLA_END")


def _ops_csv(rows):
    return pd.DataFrame(rows, columns=["OP TYPE", "OP CODE"])


def _device(code):
    return ("tt_dnn_device", code)


def _signpost(code):
    return ("signpost", code)


def test_one_region_keeps_only_its_ops():
    df = _ops_csv(
        [
            _device("Tilize"),
            _signpost("MLA_START"),
            _device("Matmul"),
            _device("RingJointSDPA"),
            _signpost("MLA_END"),
            _device("Typecast"),
        ]
    )
    assert ops_in_signposted_region(df, _REGION)["OP CODE"].tolist() == ["Matmul", "RingJointSDPA"]


@pytest.mark.parametrize(
    "rows",
    [
        # Two forwards in one profiling run, e.g. an unpinned -k axis: they would be summed.
        [
            _signpost("MLA_START"),
            _device("Matmul"),
            _signpost("MLA_END"),
            _signpost("MLA_START"),
            _device("Matmul"),
            _signpost("MLA_END"),
        ],
        # Nested regions.
        [_signpost("MLA_START"), _signpost("MLA_START"), _device("Matmul"), _signpost("MLA_END"), _signpost("MLA_END")],
        # No region.
        [_device("Matmul")],
        # A start without a stop.
        [_signpost("MLA_START"), _device("Matmul")],
    ],
    ids=["two_regions", "nested", "none", "unclosed"],
)
def test_anything_but_one_region_fails(rows, expect_error):
    with expect_error(pytest.fail.Exception, "expected one region per profiling run"):
        ops_in_signposted_region(_ops_csv(rows), _REGION)


def test_empty_region_fails(expect_error):
    df = _ops_csv([_signpost("MLA_START"), _signpost("MLA_END"), _device("Matmul")])
    with expect_error(pytest.fail.Exception, "matched no device rows"):
        ops_in_signposted_region(df, _REGION)
