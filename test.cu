#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloFromGPU() {
    printf("Hola desde el GPU! Hilo (%d, %d)\n", blockIdx.x, threadIdx.x);
}

int main() {
    printf("Hola desde la CPU!\n");

    // Lanzamos un kernel simple con 2 bloques de 4 hilos cada uno
    helloFromGPU<<<2, 4>>>();

    // Esperamos a que el kernel termine y mostramos errores si hay
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error al sincronizar GPU: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Kernel terminado sin errores.\n");
    return 0;
}
