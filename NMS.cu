#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/count.h>

// 定义边界框坐标在数组中的索引
#define X1 0
#define Y1 1
#define X2 2
#define Y2 3

__device__ float IoU(const float* A, const float* B) {
    // 计算交集
    float x_inter1 = max(A[X1], B[X1]);
    float y_inter1 = max(A[Y1], B[Y1]);
    float x_inter2 = min(A[X2], B[X2]);
    float y_inter2 = min(A[Y2], B[Y2]);
    
    // 检查是否有有效的交集
    if (x_inter2 <= x_inter1 || y_inter2 <= y_inter1)
        return 0.0f;
    
    // 计算两个框的面积
    float area_A = (A[X2] - A[X1]) * (A[Y2] - A[Y1]);
    float area_B = (B[X2] - B[X1]) * (B[Y2] - B[Y1]);
    
    // 计算交集面积
    float w = x_inter2 - x_inter1;
    float h = y_inter2 - y_inter1;
    float inter_area = w * h;
    
    // 计算并集面积
    float union_area = area_A + area_B - inter_area;
    
    if (union_area <= 0 || area_A <= 0 || area_B <= 0)
        return 0.0f;
    
    return inter_area / union_area;
}

__global__ void computerNMS(float* boxes, int* indices, int* keep, int num_boxes, float threshold, int ref_idx) {      
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boxes || idx <= ref_idx) return;  

    // 只有当当前框还未被抑制时才进行计算
    if (keep[indices[idx]] == 1) {
        float* max_box = boxes + indices[ref_idx]*4;
        float* cur_box = boxes + indices[idx]*4;
        float iou = IoU(max_box, cur_box);

        if (iou > threshold) {
            keep[indices[idx]] = 0;  // 抑制重叠框
        }
    }
}

int* NMS(float* boxes, float* confidence, int num_boxes, float threshold, int* num_keep) {
    thrust::device_vector<int> indices(num_boxes);
    thrust::sequence(indices.begin(), indices.end());

    // 排序
    thrust::sort_by_key(thrust::device, confidence, confidence + num_boxes, indices.begin(), thrust::greater<float>());

    thrust::device_vector<int> keep(num_boxes, 1);
    thrust::host_vector<int> h_keep(num_boxes, 1);

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_boxes + threadsPerBlock - 1) / threadsPerBlock;

    for (size_t i = 0; i < num_boxes; i++) {
        // 只处理未被抑制的框
        if (keep[indices[i]] == 0) continue;

        computerNMS<<<blocksPerGrid, threadsPerBlock>>>(
            boxes, thrust::raw_pointer_cast(indices.data()),
            thrust::raw_pointer_cast(keep.data()), num_boxes, threshold, i);
        
        cudaDeviceSynchronize();
    }
    
    // 计算保留的框的数量
    *num_keep = thrust::count(keep.begin(), keep.end(), 1);
    
    // 分配内存存储保留的索引
    int* keep_indices;
    cudaMalloc(&keep_indices, (*num_keep) * sizeof(int));
    
    // 创建一个临时向量存储结果
    std::vector<int> result_indices;
    // 更新h_keep为最新的keep状态
    thrust::copy(keep.begin(), keep.end(), h_keep.begin());
    
    // 使用排序后的索引顺序来收集结果
    for(int i = 0; i < num_boxes; i++) {
        if(h_keep[indices[i]] == 1) {
            result_indices.push_back(indices[i]);
        }
    }
    
    // 复制结果到GPU
    cudaMemcpy(keep_indices, result_indices.data(), 
               (*num_keep) * sizeof(int), cudaMemcpyHostToDevice);
    
    return keep_indices;
}

int main() {
    // 设置控制台支持中文输出
    system("chcp 65001");  // 设置控制台编码为 UTF-8
    
    // 测试数据：5个边界框
    float h_boxes[] = {
        100, 100, 200, 200,  // 第1个框
        120, 120, 220, 220,  // 第2个框
        130, 130, 230, 230,  // 第3个框
        125, 125, 225, 225,  // 第4个框（修改坐标使其与其他框重叠）
        120, 120, 210, 210   // 第5个框
    };
    
    float h_confidence[] = {0.9, 0.8, 0.7, 0.8, 0.75};  // 置信度
    int num_boxes = 5;
    float nms_threshold = 0.5;
    
    // 分配GPU内存
    float *d_boxes, *d_confidence;
    cudaMalloc(&d_boxes, num_boxes * 4 * sizeof(float));
    cudaMalloc(&d_confidence, num_boxes * sizeof(float));
    
    // 复制数据到GPU
    cudaMemcpy(d_boxes, h_boxes, num_boxes * 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_confidence, h_confidence, num_boxes * sizeof(float), cudaMemcpyHostToDevice);
    
    // 调用NMS
    int num_keep;
    int* keep_indices = NMS(d_boxes, d_confidence, num_boxes, nms_threshold, &num_keep);
    
    // 将结果复制回CPU
    int* h_keep_indices = (int*)malloc(num_keep * sizeof(int));
    cudaMemcpy(h_keep_indices, keep_indices, num_keep * sizeof(int), cudaMemcpyDeviceToHost);
    
    // 打印结果
    printf("\n=== NMS结果 ===\n");
    printf("保留框数量：%d\n", num_keep);
    printf("保留框索引：");
    for(int i = 0; i < num_keep; i++) {
        printf("%d ", h_keep_indices[i]);
    }
    printf("\n\n");
    
    // 释放内存
    cudaFree(d_boxes);
    cudaFree(d_confidence);
    cudaFree(keep_indices);
    free(h_keep_indices);
    
    return 0;
}