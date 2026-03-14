#include <cuda_runtime.h>

#define DIM 1000

__global__ void add_kernel(int a, int b, int *c) {
    *c = a + b;
}

struct cuComplex {
    float r;
    float i;
    __host__ __device__ cuComplex(float a, float b) : r(a), i(b) {}
    __device__ float magnitude2(void) { return r * r + i * i; }
    __device__ cuComplex operator*(const cuComplex& a) {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    __device__ cuComplex operator+(const cuComplex& a) {
        return cuComplex(r+a.r, i+a.i);
    }
};

__device__ int julia(int x, int y) {
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);
    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);
    for (int i=0; i<200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000) return 0;
    }
    return 1;
}

__global__ void julia_kernel(unsigned char *ptr) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;
    int juliaValue = julia(x, y);
    ptr[offset*4 + 0] = 255 * juliaValue;
    ptr[offset*4 + 1] = 0;
    ptr[offset*4 + 2] = 0;
    ptr[offset*4 + 3] = 255;
}

__global__ void vector_add_kernel(int *a, int *b, int *c, int N) {
    int tid = blockIdx.x;
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}

extern "C" {

void run_add(int a, int b, int *result) {
    int *dev_result;
    cudaMalloc((void**)&dev_result, sizeof(int));
    add_kernel<<<1, 1>>>(a, b, dev_result);
    cudaDeviceSynchronize();
    cudaMemcpy(result, dev_result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_result);
}

void run_julia(unsigned char *output) {
    unsigned char *dev_bitmap;
    int size = DIM * DIM * 4;
    cudaMalloc((void**)&dev_bitmap, size);
    dim3 grid(DIM, DIM);
    julia_kernel<<<grid, 1>>>(dev_bitmap);
    cudaDeviceSynchronize();
    cudaMemcpy(output, dev_bitmap, size, cudaMemcpyDeviceToHost);
    cudaFree(dev_bitmap);
}

void run_vector_add(int N, int *a, int *b, int *c) {
    int *dev_a, *dev_b, *dev_c;
    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));
    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
    vector_add_kernel<<<N, 1>>>(dev_a, dev_b, dev_c, N);
    cudaDeviceSynchronize();
    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}

}