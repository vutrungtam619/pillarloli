from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pillarpainting_ops',
    version='0.1',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='packages.voxel_op',
            sources=[
                'packages/voxelization/voxelization.cpp',
                'packages/voxelization/voxelization_cpu.cpp',
                'packages/voxelization/voxelization_cuda.cu',
            ],
            define_macros=[('WITH_CUDA', None)]
        ),
        CUDAExtension(
            name='packages.iou3d_op',
            sources=[
                'packages/iou3d/iou3d.cpp',
                'packages/iou3d/iou3d_kernel.cu',
            ],
            define_macros=[('WITH_CUDA', None)]
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False
)