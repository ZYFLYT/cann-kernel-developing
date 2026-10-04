#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
// CANN昇腾ACL头文件，Host与Device交互相关定义
#include "acl/acl.h"
// Ascend-C算子开发核心头文件：Pipe、MTE、Vector API都在这里
#include "kernel_operator.h"

// ===================== 双缓冲Buffer数量定义 =====================
// constexpr：编译期常量，编译的时候值就固定，运行不可修改
// BUFFER_NUM = 2 代表【双缓冲Double Buffer】，每个输入队列分配2块UB缓冲区
constexpr uint32_t BUFFER_NUM=2;

// ===================== Tiling分块参数结构体 =====================
/**
 * @brief Tiling结构体：Host侧计算好分块信息，传递给Device侧Kernel
 * 作用：因为AI Core的UB片上内存很小，无法一次性加载全部数据，需要把大张量切分成小块(tile)
 * Host在LaunchKernel之前，提前计算好每个AI Core需要处理多少数据，打包在这里传给Kernel
 * 注意：结构体里面只放必要参数，多余成员会降低下发性能
 */
 struct AddCustomTilingData
 {
    uint32_t totalLength; // 整个输入张量总的元素个数，本次样例固定 8*2048 个元素
    uint32_t tileNum; // 当前这一个AI Core，一共需要循环处理多少个tile（数据块）
 }

 // ===================== Kernel核函数入口 =====================
/**
 * @brief Add算子Device侧核函数入口，跑在AI Core上
 * __global__ ：标记这是顶层核函数入口，Host可以调度多个AI Core并行执行
 * __aicore__ ：标记函数运行在AI Core硬件，内部可以使用Pipe、MTE、Vector向量指令
 * @param tilingData ：Host传过来的分块参数（上面Tiling结构体）
 * @param inX ：输入X，__gm__代表指针指向NPU全局显存GM
 * @param inY ：输入Y，__gm__全局显存指针
 * @param outZ：输出Z，__gm__全局显存指针，存放 X+Y 的结果
 */

 // CANN Ascend-C 头文件 kernel_operator.h 里面定义的 
 // #define GM_ADDR  __gm__ uint8_t *

 extern "C"
__global__ __aicore__ void add_custom(GM_ADDR x,GM_ADDR y,GM_ADDR z,AddCustomTilingData tiling){
   // 设置Kernel任务类型：KERNEL_TYPE_AIV_ONLY 代表向量核，走向量计算流水线
   KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)
   // 实例化Kernel类对象op，这个类是我们自己实现的，包含Init / Process
   KernelAdd op;
   op.Init(x,y,z,tiling.totalLength,tiling.tileNum);
   op.Process();
}


// 自定义Kernel算子类，跑在AI Core上，封装整个Add算子流水线逻辑
class KernelAdd{
public:
   // 构造函数
   // __aicore__：这个函数运行在AI Core硬件上
   // inline：内联函数，编译器直接把代码嵌入调用处，减少函数调用开销，Device侧代码常用
   __aicore__ inline void KernelAdd(){}
   // 初始化函数，完成内存初始化、管道队列创建
   __aicore__ inline void Init(GM_ADDR x,GM_ADDR y,GM_ADDR z,uint32_t totalLength,uint32_t tileNum);
   // 调用私有成员CopyIn、Compute、CopyOut，实现矢量三级流水线：搬入→计算→搬出
   __aicore__ inline void Process();

private:
   // 搬入函数 CopyIn
   // 从GM全局显存搬运数据到LocalMemory（UB片上缓存）
   __aicore__ inline void CopyIn(uint32_t progress);
   __aicore__ inline void Compute(uint32_t progress);
   __aicore__ inline void CopyOut(uint32_t progress);

private:
   AscendC::TPipe pipe;// TPipe：CANN管道管理对象，用来创建流水线、管理MTE DMA搬运任务
   AscendC::TQueue<AscendC::Tposition::VECIN,BUFFER_NUM> inQueueX,inQueueY;
   // 输出队列，TPosition::VECOUT，给CopyOut用（UB搬运回GM）
   AscendC::Tqueue<AscendC::Tposition::VECOUT,BUFFER_NUM> outQueueZ;

   // GlobalTensor：CANN封装的GM全局显存张量对象，代替原始GM_ADDR指针，方便地址偏移
   AscendC::GlobalTenser<half> xGm;// 输入X，GM显存张量
   AscendC::GlobalTenser<half> yGm;
   AscendC::GlobalTenser<half> zGm;

   uint32_t blockLength; // 每个核的计算数据长度
   uint32_t tileNum;     // 当前AI Core一共要处理多少个tile小块（来自Tiling）
   uint32_t tileLength;  // 单个tile内部的元素长度
};


__aicore__ inline void KernelAdd::Init(GM_ADDR x,GM_ADDR y, GM_ADDR z,uint32_t totalLength,uint32_t tileNum){
   this->blockLength=totalLength/AscendC::GetBlockNum();
   this->tileNum=tileNum;
   this->tileLength=this->blockLength/tileNum/BUFFER_NUM;

   // ===================== 设置GlobalTensor，绑定当前核在GM显存上的数据起始位置 =====================
   // GetBlockIdx()：拿到当前AI Core编号 blockIdx（0~7）
   // this->blockLength * AscendC::GetBlockIdx()：算出当前核在全局GM上的偏移量
   // SetGlobalBuffer：给GlobalTensor绑定GM显存的起始地址 + 处理长度
   xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength*AscendC::GetBlockIndex(),this->blockLength);
   yGm.SetGlobalBuffer((__gm__ float *)y + this->blockLength*AscendC::GetBlockIndex(),this->blockLength);
   zGm.SetGlobalBuffer((__gm__ float *)z + this->blockLength*AscendC::GetBlockIndex(),this->blockLength);
   // (__gm__ float *)x：把GM_ADDR原始地址转成float类型全局显存指针

   // ===================== TPipe管道初始化，给队列分配UB片上内存 =====================
   // pipe.InitBuffer(队列对象, 缓冲区数量BUFFER_NUM, 单个缓冲区字节大小)
   // tileLength * sizeof(float)：一块UB缓冲区占多少字节，tileLength=128，float4字节 → 128*4
   pipe.InitBuffer(inQueueX,BUFFER_NUM,this->tileLength*sizeof(float));
   pipe.InitBuffer(inQueueY,BUFFER_NUM,this->tileLength*sizeof(float));
   pipe.InitBuffer(outQueueZ,BUFFER_NUM,this->tileLength*sizeof(float));
}

