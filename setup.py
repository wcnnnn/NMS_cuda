from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import numpy as np

setup(
    name='nms_cuda',
    ext_modules=[
        CUDAExtension('nms_cuda', 
            sources=['NMS2py.cu'],
            include_dirs=[np.get_include()],
            extra_compile_args={
                'cxx': [],
                'nvcc': ['-O3']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
) 