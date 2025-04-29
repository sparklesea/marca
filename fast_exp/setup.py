from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fast_exp_cuda',
    ext_modules=[
        CUDAExtension(
            name='fast_exp_cuda',
            sources=['fast_exp_kernel.cu'],
            extra_compile_args={'nvcc': ['-O3']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)