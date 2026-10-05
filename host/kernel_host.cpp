// 引入C++标准库头文件
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <cmath>

// CANN运行时API头文件，acl是昇腾底层运行库，负责设备、内存、流、拷贝
#include "acl/acl.h"
// #include "acl/aclrt.h"

// 引入Device端AscendC内核定义（Sources目录下的流水线KernelAdd）
//#include "../Sources/ascendc_kernels_bbit.cpp"

// ========== Tiling结构体：【Host和Device两边必须完全一模一样，一字不差】 ==========
// 作用：Host把分块信息传给AI Core，告诉每个核：总数据长度、单次片上处理tile块的长度
struct AddCustomTilingData{
    uint32_t totalLength;   // 整个向量全部元素数量
    uint32_t tileLength;    // 每个AI Core，一次搬运到片上LM处理的数据长度
};

/**
 * @brief  Host侧封装函数：完成NPU内存申请、数据拷贝、提交算子、回收显存/锁页内存
 * @param  x : CPU侧输入向量x
 * @param  y : CPU侧输入向量y
 * @param  stream : 外部创建好的NPU任务流
 * @return std::vector<float> : NPU算子计算得到的z = x + y
 */
 std::vector<float> kernel_add(std::vector<float> &x, std::vector<float> &y, aclrtStream stream){
    // blockDim：启动多少个AI Core并行计算，这里使用8核
    constexpr uint32_t blockDim = 8;
    uint32_t totalLength=x.size();
    size_t totalByteSize=totalLength*sizeof(float);
    int32_t deviceId=0;

    // 构造tiling分块参数，传给Device内核；tileLength=128：每次片上处理128个float
    AddCustomTilingData tiling={totalLength,128};
    // xHost/yHost：仅仅是指针别名，指向外部传入vector底层内存，不需要我们分配、不需要释放
    // reinterpret_cast：强制转换指针类型，只改变解释内存的方式，不拷贝、不移动数据
    uint8_t *xHost=reinterpret_cast<uint8_t *>(x.data());
    uint8_t *yHost=reinterpret_cast<uint8_t *>(y.data());
    // zHost：Host锁页内存，用于存放NPU计算完返回的结果；后面必须手动释放
    uint8_t *zHost=nullptr;
    // xDevice/yDevice/zDevice：NPU全局显存（Global Memory）上的指针，存放输入输出张量
    uint8_t *xDevice=nullptr;
    uint8_t *yDevice=nullptr;
    uint8_t *zDevice=nullptr;

    // ====================== 3. 分配内存 ======================
    // aclrtMallocHost：在CPU Host上分配【锁页内存 pinned memory】
    // 锁页内存不会被操作系统swap交换到磁盘，Host<->Device拷贝速度更快，专门用于NPU数据交互
    aclrtMallocHost((void **)&zHost, totalByteSize);
    // aclrtMalloc：在NPU全局显存分配内存，ACL_MEM_MALLOC_HUGE_FIRST优先大页显存，提升访存性能
    aclrtMalloc((void **)&xDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&yDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&zDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    // ====================== 4. Host CPU内存 拷贝到 NPU Device显存（异步任务） ======================
    // aclrtMemcpy：昇腾内存拷贝API，ACL_MEMCPY_HOST_TO_DEVICE：方向CPU -> NPU显存
    // 这个函数提交任务进stream，是非阻塞异步，CPU不会原地等待拷贝结束
    aclrtMemcpy(xDevice, totalByteSize, xHost, totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(yDevice, totalByteSize, yHost, totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    // ====================== 5. 提交AscendC算子到NPU执行 ======================
    // KernelAdd<<<核数量, nullptr, stream>>>(参数...);
    // <<<>>>是昇腾核调用语法，对应CUDA的<<<>>>；启动blockDim个AI Core，送入stream队列
    KernelAdd<<<blockDim, nullptr, stream>>>(xDevice,yDevice,zDevice,tiling);
    // ====================== 6. 同步等待 ======================
    // aclrtSynchronizeStream：阻塞！CPU在这里停下，一直等待stream里面【所有任务全部执行完毕】
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(zHost, totalByteSize, zDevice, totalByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    // 构造vector：从zHost锁页内存，拷贝totalLength个float，新建独立vector
    // (float *)zHost：起始地址；(float *)zHost + totalLength：结束地址（开区间，不包含）
    // 这里会做内存拷贝，新vector拥有独立数据；后面就算释放zHost内存，vector数据不受影响
    std::vector<float> z((float *)zHost, (float *)zHost + totalLength);

    // ====================== 8. 资源释放：【谁申请，谁释放；释放顺序和申请顺序反过来】 ======================
    aclrtFree(xDevice);
    aclrtFree(yDevice);
    aclrtFree(zDevice);
    aclrtFreeHost(zHost);

    return z;
 }


 /**
 * @brief VerifyResult 算子结果校验函数：比对NPU输出output 和 CPU标准golden真值
 * @param output : NPU算子跑出来的结果
 * @param golden : CPU计算的标准正确结果（真值）
 * @return uint32_t : 返回0=校验成功，返回1=校验失败
 */
uint32_t VerifyResult(std::vector<float> &output, std::vector<float> &golden)
{
    // 定义lambda匿名函数：打印tensor前20个元素，用于查看数据
    auto printTensor = [](std::vector<float> &tensor, const char *name) {
        // constexpr 编译期常量，最多打印20个元素
        constexpr size_t maxPrintSize = 20;
        std::cout << name << ": ";
        // 取tensor开始 到 min(数组全长,20)位置的迭代器
        std::copy(tensor.begin(), tensor.begin() + std::min(tensor.size(), maxPrintSize),
                  std::ostream_iterator<float>(std::cout, ", "));
        // 如果向量长度大于20，打印省略号，代表后面还有数据
        if (tensor.size() > maxPrintSize)
        {
            std::cout << "...";
        }
        std::cout << std::endl;
    };

    // 调用lambda，打印NPU输出、CPU真值，前20个元素
    printTensor(output, "Output");
    printTensor(golden, "Golden");

    // ========= 修改：浮点数误差比对，不是严格相等 =========
    if (output.size() != golden.size())
    {
        std::cout << "[Failed] Case accuracy is verification failed! Size mismatch." << std::endl;
        return 1;
    }
    bool all_ok = true;
    for(size_t i = 0; i < output.size(); i++)
    {
        if(fabs(output[i] - golden[i]) > 1e-5)
        {
            all_ok = false;
            break;
        }
    }
    if(all_ok)
    {
        std::cout << "[Success] Case accuracy is verification passed." << std::endl;
        return 0; // 返回0，代表成功
    }
    else
    {
        std::cout << "[Failed] Case accuracy is verification failed!" << std::endl;
        return 1; // 返回1，代表校验失败
    }
    return 0;
}


/**
 * @brief main 主函数：程序入口，生成测试数据，调用算子，生成golden真值，调用校验函数
 * @param argc 参数个数，argv参数数组
 * @return int32_t 程序退出码，0成功，非0失败
 */
int32_t main(int32_t argc, char *argv[])
{
    // ========= aclInit全局初始化，只在main最开始执行一次 =========
    aclInit(nullptr);
    int32_t deviceId = 0;
    aclrtSetDevice(deviceId);
    aclrtStream stream=nullptr;
    aclrtCreateStream(&stream);

    // 总数据长度：8 * 2048 = 16384个float元素
    constexpr uint32_t totalLength = 8 * 2048;
    // 定义x、y里面每个元素的值
    constexpr float valueX = 1.2f;
    constexpr float valueY = 2.3f;

    // 构造向量x，全部元素填充 valueX=1.2
    std::vector<float> x(totalLength, valueX);
    // 构造向量y，全部元素填充 valueY=2.3
    std::vector<float> y(totalLength, valueY);

    // 调用我们写好的Host封装函数kernel_add，在NPU上跑x+y，拿到算子输出output
    std::vector<float> output = kernel_add(x, y, stream);

    // golden：CPU上直接计算标准结果，每个元素 = valueX + valueY = 3.5，作为真值
    std::vector<float> golden(totalLength, valueX + valueY);

    // 调用校验函数，比对output和golden，返回校验结果作为程序退出码
    int ret = VerifyResult(output, golden);

    // ========= 释放资源，main结束前统一清理 =========
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return ret;
}