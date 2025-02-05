# CUDA加速的非极大值抑制(NMS)实现

本项目实现了基于CUDA的非极大值抑制(Non-Maximum Suppression, NMS)算法，并与Python原生实现进行了性能对比。

## 环境要求

- CUDA 12.4
- PyTorch 2.5.1
- Python 3.8+
- pybind11

## 性能测试结果

在不同数量的边界框下进行测试，CUDA版本相比Python原生实现取得了显著的性能提升：

| 框数量 | CUDA时间(ms) | Python时间(ms) | 加速比 |
|--------|--------------|----------------|---------|
| 100    | 0.408       | 6.394          | 15.68x  |
| 1000   | 0.804       | 31.068         | 38.62x  |
| 10000  | 1.529       | 161.314        | 105.50x |

## 项目结构

- `NMS.cu`: CUDA实现的NMS核心算法
- `benchmark.py`: 性能测试脚本
- `setup.py`: Python包构建配置
- `build.bat`: Windows下的编译脚本
- `.vscode/`: VSCode配置文件目录
  - `launch.json`: 调试配置
  - `c_cpp_properties.json`: C/C++和CUDA编译器配置
  - `tasks.json`: 构建任务配置
  - `settings.json`: VSCode编辑器设置

## VSCode配置说明

本项目包含了完整的VSCode配置，支持CUDA C++的开发和调试：

1. **编译器配置 (c_cpp_properties.json)**:
   - 使用CUDA 12.4编译器
   - 包含了必要的CUDA和PyTorch头文件路径
   - 支持C++17标准
   - 配置了适当的宏定义

2. **调试配置 (launch.json)**:
   - 支持普通C++程序调试
   - 支持CUDA程序调试（使用cuda-gdb）
   - 配置了调试器路径和启动参数

3. **任务配置 (tasks.json)**:
   - 包含CUDA代码的编译任务
   - 支持自动构建和调试

要使用这些配置，请确保：
- 已安装VSCode的C++和CUDA扩展
- CUDA工具链路径正确配置
- PyTorch开发环境正确设置

运行方法:
1. 运行Run task->CUDA build and run编译并运行NMS.cu进行测试
2. python setup.py install 安装NMS2py.cu的python扩展
3. python benchmark.py 运行性能测试


