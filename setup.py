# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import glob

from setuptools import find_packages, setup

from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

library_name = "hpco"

if torch.__version__ >= "2.6.0":
    py_limited_api = True
else:
    py_limited_api = False


def find_files(root_dir: str, ext=".cc", ignore_paths: list = None) -> list:
    if not os.path.isdir(root_dir):
        print(f"错误: '{root_dir}' 不是一个有效的目录。")
        return []

    files = []

    # 将ignore_paths转换为绝对路径集合，方便后续判断
    # 并且处理路径分隔符的兼容性
    normalized_ignore_paths = set()
    if ignore_paths:
        for p in ignore_paths:
            # 确保忽略路径也是绝对路径，并标准化，便于比较
            abs_p = os.path.abspath(p)
            # 确保路径以分隔符结尾，以便正确匹配子目录
            if not abs_p.endswith(os.sep):
                abs_p += os.sep
            normalized_ignore_paths.add(abs_p)

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 将当前目录标准化为绝对路径，并以分隔符结尾
        current_abs_dirpath = os.path.abspath(dirpath)
        if not current_abs_dirpath.endswith(os.sep):
            current_abs_dirpath += os.sep

        # 检查当前目录是否在忽略列表中，或是否是忽略路径的子目录
        should_ignore_current_dir = False
        for ignore_p in normalized_ignore_paths:
            # 如果当前目录是某个忽略路径，或者当前目录是某个忽略路径的子目录，则忽略
            if current_abs_dirpath.startswith(ignore_p):
                should_ignore_current_dir = True
                break

        if should_ignore_current_dir:
            # 如果当前目录被忽略，则跳过此目录及其子目录的遍历
            # 通过清空dirnames来告诉os.walk不要进入子目录
            dirnames[:] = []
            continue

        # 查找当前目录下的 .cu 文件
        for filename in filenames:
            if filename.endswith(ext):
                files.append(os.path.join(dirpath, filename))

    return files


def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    use_cuda = os.getenv("USE_CUDA", "1") == "1"
    if debug_mode:
        print("Compiling in debug mode")

    use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
            "-DPy_LIMITED_API=0x03090000",  # min CPython version 3.9
        ],
        "nvcc": [
            "-O3" if not debug_mode else "-O0",
        ],
    }
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, library_name, "csrc")
    sources = find_files(extensions_dir)

    cuda_sources = find_files(extensions_dir, ".cu")

    if use_cuda:
        sources += cuda_sources

    ext_modules = [
        extension(
            f"{library_name}._C",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            py_limited_api=py_limited_api,
        )
    ]

    return ext_modules


setup(
    name=library_name,
    version="0.0.1",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=["torch"],
    description="Example of PyTorch C++ and CUDA extensions",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pytorch/extension-cpp",
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
)
