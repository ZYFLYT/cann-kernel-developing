#!/bin/bash
set -e
# 加载CANN环境（根据你服务器实际CANN版本）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 源码路径
HOST_SRC=host/kernel_host.cpp
KERNEL_SRC=Sources/ascendc_kernels_bbit.cpp
OUT=output/add_custom

# 创建输出目录
mkdir -p output

# bisheng异构编译，同时传入host和kernel源码
bisheng ${HOST_SRC} ${KERNEL_SRC} -o ${OUT} --soc_version=Ascend310P3
# ↑ soc_version改成你NPU对应的型号，npu-smi info可以查到