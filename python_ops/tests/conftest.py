import pytest
import os
import importlib.util


def is_package_installed(package_name):
    """检查指定的包是否可以导入"""
    return importlib.util.find_spec(package_name) is not None


# --- 条件标记定义 ---

# 检查 triton 是否安装
triton_installed = is_package_installed("triton")

# 检查 tilelang 是否安装
tilelang_installed = is_package_installed("tilelang")

# 定义 pytest 标记，用于条件跳过
skip_if_triton_missing = pytest.mark.skipif(
    not triton_installed, reason="需要安装 triton 包才能运行此测试"
)
force_skip_triton = os.environ.get("DISABLE_TRITON_TESTS", "0") == "1"

skip_if_tilelang_missing = pytest.mark.skipif(
    not tilelang_installed, reason="需要安装 tilelang 包才能运行此测试"
)

should_run_triton_tests = is_package_installed("triton") and not force_skip_triton
skip_if_triton_missing = pytest.mark.skipif(
    not should_run_triton_tests,  # 使用新的综合条件
    reason="Triton 包缺失或通过环境变量手动禁用（DISABLE_TRITON_TESTS=1）",
)


# 注册自定义标记（可选，但推荐）
def pytest_configure(config):
    config.addinivalue_line("markers", "triton_test: 标记需要 triton 的测试")
    config.addinivalue_line("markers", "tilelang_test: 标记需要 tilelang 的测试")
