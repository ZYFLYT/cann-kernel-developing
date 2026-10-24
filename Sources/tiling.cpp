#include <cmath>
#include "kernel_operator.h"

// ==========================================
// 1. 工业级 Tiling 结构体
// ==========================================
struct TilingDataTemplate {
    uint32_t smallCoreDataNum;   // 小核处理的总元素个数
    uint32_t bigCoreDataNum;     // 大核处理的总元素个数
    uint32_t finalBigTileNum;    // 大核循环次数
    uint32_t finalSmallTileNum;  // 小核循环次数
    uint32_t tileDataNum;        // 单次最大搬运元素个数
    uint32_t smallTailDataNum;   // 小核最后一次循环处理个数
    uint32_t bigTailDataNum;     // 大核最后一次循环处理个数
    uint32_t tailBlockNum;       // 大核的个数
};

// ==========================================
// 2. Device 侧核函数（严格配对 Alloc/EnQue/DeQue/Free）
// ==========================================
__global__ __vector__ void add_custom(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, TilingDataTemplate tiling)
{
    uint32_t blockIdx = AscendC::GetBlockIdx();
    uint32_t offset = 0;
    uint32_t currentCoreDataNum = 0;
    uint32_t loopCount = 0;
    uint32_t lastTileDataNum = 0;

    // 判断当前是大核还是小核，并计算偏移量
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

    // 初始化流水线 (单缓冲模式)
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX1, inQueueX2;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    
    pipe.InitBuffer(inQueueX1, 1, tiling.tileDataNum * sizeof(float));
    pipe.InitBuffer(inQueueX2, 1, tiling.tileDataNum * sizeof(float));
    pipe.InitBuffer(outQueueY, 1, tiling.tileDataNum * sizeof(float));

    // 开始循环 (严格流水线)
    for (uint32_t i = 0; i < loopCount; ++i) {
        uint32_t tileOffset = i * tiling.tileDataNum;
        uint32_t currentTileLength = (i == loopCount - 1) ? lastTileDataNum : tiling.tileDataNum;

        if (currentTileLength <= 0) continue;

        // ================= 搬入阶段 =================
        AscendC::LocalTensor<float> x1Local = inQueueX1.AllocTensor<float>();
        AscendC::LocalTensor<float> x2Local = inQueueX2.AllocTensor<float>();
        
        AscendC::DataCopy(x1Local, x1Gm[tileOffset], currentTileLength);
        AscendC::DataCopy(x2Local, x2Gm[tileOffset], currentTileLength);
        
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);

        // ================= 计算阶段 =================
        // 正确使用 DeQue 等待数据就绪
        AscendC::LocalTensor<float> x1Ready = inQueueX1.DeQue<float>();
        AscendC::LocalTensor<float> x2Ready = inQueueX2.DeQue<float>();
        
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        
        // 执行加法
        AscendC::Add(yLocal, x1Ready, x2Ready, currentTileLength);
        
        outQueueY.EnQue(yLocal);
        
        // 释放输入
        inQueueX1.FreeTensor(x1Ready);
        inQueueX2.FreeTensor(x2Ready);

        // ================= 搬出阶段 =================
        // 正确使用 DeQue 等待计算完成
        AscendC::LocalTensor<float> yReady = outQueueY.DeQue<float>();
        
        AscendC::DataCopy(yGm[tileOffset], yReady, currentTileLength);
        
        outQueueY.FreeTensor(yReady);
    }
}

// ==========================================
// 3. Host 侧入口（最稳妥的元素级大小核均分）
// ==========================================
extern "C" void run_kernel(
    GM_ADDR x1, const TensorGroupInfo& info_x1, 
    GM_ADDR x2, const TensorGroupInfo& info_x2, 
    GM_ADDR y,  const TensorGroupInfo& info_y, 
    int64_t availableCoreNum, 
    aclrtStream stream)
{
    // 计算总元素个数
    uint32_t totalLength = 1;
    for (int i = 0; i < info_x1.tensors[0].numDims; ++i) {
        totalLength *= info_x1.tensors[0].shape[i];
    }
    if (totalLength == 0) return;

    // 确定核数与单次搬运量
    uint32_t blockNum = availableCoreNum;
    if (blockNum > totalLength) blockNum = totalLength; // 数据太少，不用开那么多核
    
    uint32_t tileDataNum = 4096; // 保守值，绝对不爆 UB

    // 按元素均分，完美避开尾块越界问题
    uint32_t everyCoreDataNum = totalLength / blockNum; // 小核拿的基础量
    uint32_t tailCoreNum = totalLength % blockNum;      // 大核数量（余数分给前面）

    TilingDataTemplate tiling;
    tiling.tailBlockNum = tailCoreNum;
    tiling.tileDataNum = tileDataNum;

    // 计算大核参数（多拿 1 个元素）
    tiling.bigCoreDataNum = everyCoreDataNum + 1;
    tiling.finalBigTileNum = (tiling.bigCoreDataNum + tileDataNum - 1) / tileDataNum;
    tiling.bigTailDataNum = tiling.bigCoreDataNum % tileDataNum;
    if (tiling.bigTailDataNum == 0) tiling.bigTailDataNum = tileDataNum;

    // 计算小核参数
    tiling.smallCoreDataNum = everyCoreDataNum;
    if (tiling.smallCoreDataNum > 0) {
        tiling.finalSmallTileNum = (tiling.smallCoreDataNum + tileDataNum - 1) / tileDataNum;
        tiling.smallTailDataNum = tiling.smallCoreDataNum % tileDataNum;
        if (tiling.smallTailDataNum == 0) tiling.smallTailDataNum = tileDataNum;
    } else {
        tiling.finalSmallTileNum = 0;
        tiling.smallTailDataNum = 0;
    }

    // 启动核函数
    add_custom<<<blockNum, nullptr, stream>>>(x1, x2, y, tiling);
}
