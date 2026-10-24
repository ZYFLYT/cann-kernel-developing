#include <cmath>
#include "kernel_operator.h"

// ==========================================
// 1. 工业级 Tiling 结构体（满血版）
// ==========================================
struct TilingDataTemplate {
    uint32_t smallCoreDataNum;   // 小核处理的总元素个数
    uint32_t bigCoreDataNum;     // 大核处理的总元素个数
    uint32_t finalBigTileNum;    // 大核循环次数
    uint32_t finalSmallTileNum;  // 小核循环次数
    uint32_t tileDataNum;        // 单次最大搬运元素个数（由UB容量决定）
    uint32_t smallTailDataNum;   // 小核最后一次循环处理个数
    uint32_t bigTailDataNum;     // 大核最后一次循环处理个数
    uint32_t tailBlockNum;       // 大核的个数
};

// ==========================================
// 2. Device 侧核函数（完美处理大小核与尾块）
// ==========================================
__global__ __vector__ void add_custom(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, TilingDataTemplate tiling)
{
    uint32_t blockIdx = AscendC::GetBlockIdx();
    uint32_t offset = 0;
    uint32_t currentCoreDataNum = 0;
    uint32_t loopCount = 0;
    uint32_t lastTileDataNum = 0;

    // 判断当前是大核还是小核
    if (blockIdx < tiling.tailBlockNum) {
        // 我是大核
        offset = blockIdx * tiling.bigCoreDataNum;
        currentCoreDataNum = tiling.bigCoreDataNum;
        loopCount = tiling.finalBigTileNum;
        lastTileDataNum = tiling.bigTailDataNum;
    } else {
        // 我是小核
        offset = (tiling.tailBlockNum * tiling.bigCoreDataNum) + 
                 ((blockIdx - tiling.tailBlockNum) * tiling.smallCoreDataNum);
        currentCoreDataNum = tiling.smallCoreDataNum;
        loopCount = tiling.finalSmallTileNum;
        lastTileDataNum = tiling.smallTailDataNum;
    }

    if (currentCoreDataNum == 0) return;

    // 绑定全局内存
    AscendC::GlobalTensor<float> x1Gm, x2Gm, yGm;
    x1Gm.SetGlobalBuffer((__gm__ float*)x1 + offset, currentCoreDataNum);
    x2Gm.SetGlobalBuffer((__gm__ float*)x2 + offset, currentCoreDataNum);
    yGm.SetGlobalBuffer((__gm__ float*)y + offset, currentCoreDataNum);

    // 初始化流水线
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX1, inQueueX2;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    
    pipe.InitBuffer(inQueueX1, 1, tiling.tileDataNum * sizeof(float));
    pipe.InitBuffer(inQueueX2, 1, tiling.tileDataNum * sizeof(float));
    pipe.InitBuffer(outQueueY, 1, tiling.tileDataNum * sizeof(float));

    // 开始循环
    for (uint32_t i = 0; i < loopCount; ++i) {
        uint32_t tileOffset = i * tiling.tileDataNum;
        // 判断是不是最后一次循环（如果是，用尾块长度）
        uint32_t currentTileLength = (i == loopCount - 1) ? lastTileDataNum : tiling.tileDataNum;

        if (currentTileLength <= 0) continue;

        // 搬入
        AscendC::LocalTensor<float> x1Local = inQueueX1.AllocTensor<float>();
        AscendC::LocalTensor<float> x2Local = inQueueX2.AllocTensor<float>();
        AscendC::DataCopy(x1Local, x1Gm[tileOffset], currentTileLength);
        AscendC::DataCopy(x2Local, x2Gm[tileOffset], currentTileLength);
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);

        // 计算
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::Add(yLocal, x1Local, x2Local, currentTileLength);
        outQueueY.EnQue(yLocal);

        // 释放输入
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);

        // 搬出
        AscendC::LocalTensor<float> yOutLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[tileOffset], yOutLocal, currentTileLength);
        outQueueY.FreeTensor(yLocal); // 注意这里释放的是 yOutLocal 对应的 Tensor
    }
}

// ==========================================
// 3. Host 侧入口（实现截图里的核心计算逻辑）
// ==========================================
extern "C" void run_kernel(
    GM_ADDR x1, const TensorGroupInfo& info_x1, 
    GM_ADDR x2, const TensorGroupInfo& info_x2, 
    GM_ADDR y,  const TensorGroupInfo& info_y, 
    int64_t availableCoreNum, 
    aclrtStream stream)
{
    // 1. 获取总元素个数
    uint32_t totalLength = 1;
    for (int i = 0; i < info_x1.tensors[0].numDims; ++i) {
        totalLength *= info_x1.tensors[0].shape[i];
    }
    if (totalLength == 0) return;

    // 2. 设定单次处理的元素个数（保守设定，防止 UB 越界）
    // 假设浮点型，一个块 256 个元素
    uint32_t tileDataNum = 4096; 

    // 3. 计算总共需要多少块（向上取整）
    uint32_t totalBlockNum = (totalLength + tileDataNum - 1) / tileDataNum;

    // 4. 计算实际使用的核数
    uint32_t blockNum = availableCoreNum;
    if (blockNum > totalBlockNum) {
        blockNum = totalBlockNum; // 数据太少，不用开那么多核
    }

    // 5. 计算大小核分配
    uint32_t everyCoreBlockNum = totalBlockNum / blockNum; // 基础块数
    uint32_t tailBlockNum = totalBlockNum % blockNum;      // 余数（大核数量）

    TilingDataTemplate tiling;
    tiling.tailBlockNum = tailBlockNum;
    tiling.tileDataNum = tileDataNum;

    // 计算大核参数
    tiling.bigCoreDataNum = (everyCoreBlockNum + 1) * tileDataNum;
    tiling.finalBigTileNum = everyCoreBlockNum + 1;
    tiling.bigTailDataNum = tileDataNum; // 大核全满，无空余（数据量不对齐的情况暂不深入处理）

    // 计算小核参数
    if (blockNum > tailBlockNum) {
        tiling.smallCoreDataNum = everyCoreBlockNum * tileDataNum;
        tiling.finalSmallTileNum = everyCoreBlockNum;
        tiling.smallTailDataNum = tileDataNum;
    } else {
        tiling.smallCoreDataNum = 0;
        tiling.finalSmallTileNum = 0;
        tiling.smallTailDataNum = 0;
    }

    // 6. 启动核函数
    add_custom<<<blockNum, nullptr, stream>>>(x1, x2, y, tiling);
}
