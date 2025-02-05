# type: ignore
import numpy as np
import time
import torch
import nms_cuda  

# Python版本的NMS实现
def nms_python(boxes, scores, iou_threshold):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    _, order = scores.sort(0, descending=True)
    keep = []
    
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
        i = order[0]
        keep.append(i.item())

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr <= iou_threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]
    
    return torch.tensor(keep, dtype=torch.int32)

# 测试数据
def generate_test_data(num_boxes):
    np.random.seed(42)
    boxes = []
    for _ in range(num_boxes):
        x1 = np.random.randint(0, 500)
        y1 = np.random.randint(0, 500)
        w = np.random.randint(50, 100)
        h = np.random.randint(50, 100)
        boxes.append([x1, y1, x1+w, y1+h])
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.random.rand(num_boxes).astype(np.float32)
    return boxes, scores

def benchmark_all(num_boxes, num_runs=100):
    boxes, scores = generate_test_data(num_boxes)
    boxes_tensor = torch.from_numpy(boxes)
    scores_tensor = torch.from_numpy(scores)
    
    # 预热
    _ = nms_cuda.nms_cuda(boxes, scores, 0.5)
    _ = nms_python(boxes_tensor, scores_tensor, 0.5)
    
    # CUDA版本测试
    cuda_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = nms_cuda.nms_cuda(boxes, scores, 0.5)
        cuda_times.append(time.perf_counter() - start)
    
    # Python版本测试
    python_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = nms_python(boxes_tensor, scores_tensor, 0.5)
        python_times.append(time.perf_counter() - start)
    
    # 打印结果
    print(f"\n=== 性能测试结果 (框数量: {num_boxes}) ===")
    print(f"CUDA   版本平均时间: {np.mean(cuda_times)*1000:.3f} ms")
    print(f"Python 版本平均时间: {np.mean(python_times)*1000:.3f} ms")
    print(f"加速比: {np.mean(python_times)/np.mean(cuda_times):.2f}x")

if __name__ == "__main__":
    # 测试不同规模的数据
    for num_boxes in [100, 1000, 10000]:
        benchmark_all(num_boxes) 
