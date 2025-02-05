#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/count.h>
#include <Python.h>
#include <numpy/arrayobject.h>

// 定义边界框坐标在数组中的索引
#define X1 0
#define Y1 1
#define X2 2
#define Y2 3

// 定义常量
#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024
#define MAX_SHARED_MEM 48000  // 48KB

// 共享内存中的box结构
struct alignas(16) SharedBox {
    float x1, y1, x2, y2;
};

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

    if (keep[indices[idx]] == 1) {
        float* max_box = boxes + indices[ref_idx]*4;
        float* cur_box = boxes + indices[idx]*4;
        float iou = IoU(max_box, cur_box);

        if (iou > threshold) {
            keep[indices[idx]] = 0;
        }
    }
}

template<int BLOCK_SIZE>
__global__ void nms_kernel(
    const float4* boxes,    // 使用float4进行向量化加载
    const float* scores,
    const int* order,
    int* keep,
    const int n_boxes,
    const float thresh
) {
    // 分配共享内存
    __shared__ SharedBox shared_boxes[BLOCK_SIZE];
    __shared__ int shared_order[BLOCK_SIZE];
    
    const int row_start = blockIdx.y;
    const int tid = threadIdx.x;
    const int row_size = min(BLOCK_SIZE, n_boxes - row_start * BLOCK_SIZE);
    
    // 向量化加载到共享内存
    if (tid < row_size) {
        const int row_box_idx = order[row_start * BLOCK_SIZE + tid];
        float4 box = boxes[row_box_idx];
        shared_boxes[tid] = {box.x, box.y, box.z, box.w};
        shared_order[tid] = row_box_idx;
    }
    __syncthreads();
    
    // 处理当前块的所有框
    for (int i = 0; i < row_size; i++) {
        const int col_start = blockIdx.x * BLOCK_SIZE + tid;
        if (col_start >= n_boxes) continue;
        
        const int row_box_idx = shared_order[i];
        const int col_box_idx = order[col_start];
        
        if (col_start <= row_start * BLOCK_SIZE + i || !keep[col_box_idx]) continue;
        
        // 从共享内存读取框
        const SharedBox& cur_box = shared_boxes[i];
        const float4 other_box = boxes[col_box_idx];
        
        // 计算IoU
        float x1 = max(cur_box.x1, other_box.x);
        float y1 = max(cur_box.y1, other_box.y);
        float x2 = min(cur_box.x2, other_box.z);
        float y2 = min(cur_box.y2, other_box.w);
        
        float inter_area = max(0.0f, x2 - x1) * max(0.0f, y2 - y1);
        float cur_area = (cur_box.x2 - cur_box.x1) * (cur_box.y2 - cur_box.y1);
        float other_area = (other_box.z - other_box.x) * (other_box.w - other_box.y);
        float union_area = cur_area + other_area - inter_area;
        
        if (union_area > 0) {
            float iou = inter_area / union_area;
            if (iou > thresh) {
                keep[col_box_idx] = 0;
            }
        }
    }
}

// 使用CUDA流进行并行处理
static PyObject* nms_cuda(PyObject* self, PyObject* args) {
    PyArrayObject *boxes_array, *scores_array;
    float threshold;
    
    if (!PyArg_ParseTuple(args, "O!O!f", 
                         &PyArray_Type, &boxes_array,
                         &PyArray_Type, &scores_array,
                         &threshold)) {
        return NULL;
    }
    
    const int n_boxes = PyArray_DIM(boxes_array, 0);
    
    // 创建CUDA流
    const int n_streams = 4;
    cudaStream_t streams[n_streams];
    for (int i = 0; i < n_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // 分配GPU内存
    float4 *d_boxes;
    float *d_scores;
    int *d_order, *d_keep;
    
    cudaMalloc(&d_boxes, n_boxes * sizeof(float4));
    cudaMalloc(&d_scores, n_boxes * sizeof(float));
    cudaMalloc(&d_order, n_boxes * sizeof(int));
    cudaMalloc(&d_keep, n_boxes * sizeof(int));
    
    // 使用流进行异步数据传输
    cudaMemcpyAsync(d_boxes, PyArray_DATA(boxes_array), 
                   n_boxes * 4 * sizeof(float), cudaMemcpyHostToDevice, 
                   streams[0]);
    cudaMemcpyAsync(d_scores, PyArray_DATA(scores_array), 
                   n_boxes * sizeof(float), cudaMemcpyHostToDevice, 
                   streams[1]);
    
    // 初始化
    thrust::sequence(thrust::cuda::par.on(streams[2]), d_order, d_order + n_boxes);
    thrust::fill(thrust::cuda::par.on(streams[2]), d_keep, d_keep + n_boxes, 1);
    
    // 排序
    thrust::sort_by_key(thrust::cuda::par.on(streams[2]),
        d_scores, d_scores + n_boxes,
        d_order,
        thrust::greater<float>()
    );
    
    // 动态选择线程块大小
    const int block_size = min(MAX_THREADS_PER_BLOCK, 
                             (int)(sqrt(n_boxes * sizeof(SharedBox)) / 16) * 16);
    const int grid_size = (n_boxes + block_size - 1) / block_size;
    dim3 blocks(grid_size, grid_size);
    dim3 threads(block_size);
    
    // 根据不同的block_size调用不同的模板实例
    switch (block_size) {
        case 64:
            nms_kernel<64><<<blocks, threads, 0, streams[3]>>>(
                d_boxes, d_scores, d_order, d_keep, n_boxes, threshold);
            break;
        case 128:
            nms_kernel<128><<<blocks, threads, 0, streams[3]>>>(
                d_boxes, d_scores, d_order, d_keep, n_boxes, threshold);
            break;
        case 256:
            nms_kernel<256><<<blocks, threads, 0, streams[3]>>>(
                d_boxes, d_scores, d_order, d_keep, n_boxes, threshold);
            break;
        default:
            nms_kernel<512><<<blocks, threads, 0, streams[3]>>>(
                d_boxes, d_scores, d_order, d_keep, n_boxes, threshold);
    }
    
    // 获取结果
    int* h_keep = new int[n_boxes];
    cudaMemcpyAsync(h_keep, d_keep, n_boxes * sizeof(int), 
                   cudaMemcpyDeviceToHost, streams[3]);
    
    // 同步所有流
    for (int i = 0; i < n_streams; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    
    // 收集结果
    std::vector<int> keep_indices;
    for (int i = 0; i < n_boxes; i++) {
        if (h_keep[i]) {
            keep_indices.push_back(i);
        }
    }
    
    // 创建返回值
    npy_intp dims[1] = {(npy_intp)keep_indices.size()};
    PyObject* result = PyArray_SimpleNew(1, dims, NPY_INT);
    memcpy(PyArray_DATA((PyArrayObject*)result), 
           keep_indices.data(), 
           keep_indices.size() * sizeof(int));
    
    // 清理资源
    cudaFree(d_boxes);
    cudaFree(d_scores);
    cudaFree(d_order);
    cudaFree(d_keep);
    delete[] h_keep;
    
    for (int i = 0; i < n_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    
    return result;
}

// 模块方法定义
static PyMethodDef NMSMethods[] = {
    {"nms_cuda", nms_cuda, METH_VARARGS, "Execute NMS on CUDA."},
    {NULL, NULL, 0, NULL}
};

// 模块定义
static struct PyModuleDef nmsmodule = {
    PyModuleDef_HEAD_INIT,
    "nms_cuda",
    NULL,
    -1,
    NMSMethods
};

// 模块初始化函数
PyMODINIT_FUNC PyInit_nms_cuda(void) {
    import_array();  // 必须调用，初始化NumPy
    return PyModule_Create(&nmsmodule);
}
