#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define BLOCK_W 16
#define BLOCK_H 16
#define TILE_W (BLOCK_W + 2) // + halo
#define TILE_H (BLOCK_H + 2)

float kernel_1[9] = {1.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0};
float kernel_2[9] = {-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0};

// OK
void ReadBIN(float **img, int *L, int *M, int *N, int *P, const char *dimfile, const char *binfile)
{
    FILE *fp = fopen(dimfile, "r");
    if (!fp)
    {
        fprintf(stderr, "No se pudo abrir archivo de dimensiones.\n");
        exit(EXIT_FAILURE);
    }

    fscanf(fp, "%d %d %d %d\n", L, M, N, P); // Leer L M N P
    fclose(fp);

    size_t total_size = (size_t)(*L) * (*M) * (*N);
    float *img_new = new float[total_size];

    FILE *bin = fopen(binfile, "rb");
    if (!bin)
    {
        fprintf(stderr, "No se pudo abrir archivo binario de imagen.\n");
        exit(EXIT_FAILURE);
    }

    unsigned char *temp_data = new unsigned char[total_size];
    fread(temp_data, sizeof(unsigned char), total_size, bin);
    fclose(bin);

    for (size_t i = 0; i < total_size; i++)
    {
        img_new[i] = static_cast<float>(temp_data[i]); // Convertir a float para procesar
    }

    delete[] temp_data;
    *img = img_new;
}

void WriteBIN(float *imgs, int L, int M, int N, const char *dimfile_out, const char *binfile_out)
{
    FILE *fp = fopen(dimfile_out, "w");
    fprintf(fp, "%d %d %d %d\n", L, M, N, 0); // No hay padding ahora
    fclose(fp);

    size_t total_size = (size_t)L * M * N;
    unsigned char *output = new unsigned char[total_size];

    for (size_t i = 0; i < total_size; ++i)
        output[i] = (unsigned char)(imgs[i] > 255.0f ? 255 : imgs[i]);

    FILE *out = fopen(binfile_out, "wb");
    fwrite(output, sizeof(unsigned char), total_size, out);
    fclose(out);

    delete[] output;
}
void funcionCPU(float *img, float *out, int L, int M, int N)
{
    int padded_size = M * N;
    int output_size = (M - 2) * (N - 2);

    for (int l = 0; l < L; ++l) // L frames
    {
        float *frame_in = img + l * padded_size;
        float *frame_out = out + l * output_size;

        for (int i = 1; i < M - 1; ++i) // filas
        {
            for (int j = 1; j < N - 1; ++j) // columnas
            {
                float temp1 = 0.0f;
                float temp2 = 0.0f;

                for (int k1 = -1; k1 <= 1; ++k1)
                {
                    for (int k2 = -1; k2 <= 1; ++k2)
                    {
                        int img_idx = (i + k1) * N + (j + k2);
                        int k_idx = (k1 + 1) * 3 + (k2 + 1);

                        temp1 += frame_in[img_idx] * kernel_1[k_idx];
                        temp2 += frame_in[img_idx] * kernel_2[k_idx];
                    }
                }

                float grad = sqrtf(temp1 * temp1 + temp2 * temp2);
                grad = grad > 255.0f ? 255.0f : grad;

                frame_out[(i - 1) * (N - 2) + (j - 1)] = grad;
            }
        }
    }
}

__global__ void kernelNaive(float *imgs, float *out, float *kernel_1, float *kernel_2, int L, int M, int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int imsize = M * N;

    int total_threads = L * imsize;

    if (tid >= total_threads)
        return;

    int l = tid / imsize; // índice de la imagen actual
    int pixel_idx = tid % imsize;

    int i = pixel_idx / N; // fila dentro de la imagen
    int j = pixel_idx % N; // columna dentro de la imagen

    if (i > 0 && i < M - 1 && j > 0 && j < N - 1)
    {
        float temp1 = 0.0f;
        float temp2 = 0.0f;

        for (int k1 = -1; k1 <= 1; ++k1)
        {
            for (int k2 = -1; k2 <= 1; ++k2)
            {
                int img_idx = l * imsize + (i + k1) * N + (j + k2);
                int k_idx = (k1 + 1) * 3 + (k2 + 1);
                temp1 += imgs[img_idx] * kernel_1[k_idx];
                temp2 += imgs[img_idx] * kernel_2[k_idx];
            }
        }

        float grad = sqrtf(temp1 * temp1 + temp2 * temp2);
        grad = grad > 255.0f ? 255.0f : grad;

        int out_idx = l * (M - 2) * (N - 2) + (i - 1) * (N - 2) + (j - 1);
        out[out_idx] = grad;
    }
}
__global__ void kernelOpt(float *imgs, float *out, float *kernel_1, float *kernel_2, int L, int M, int N)
{
    const int TILE_WIDTH = 16;
    const int SHARED_WIDTH = TILE_WIDTH + 2;

    __shared__ float tile[SHARED_WIDTH][SHARED_WIDTH];
    __shared__ float kernel1_shared[9];
    __shared__ float kernel2_shared[9];

    // Total de tiles por imagen
    int tiles_per_row = (N + TILE_WIDTH - 1) / TILE_WIDTH;
    int tiles_per_col = (M + TILE_WIDTH - 1) / TILE_WIDTH;
    int tiles_per_image = tiles_per_row * tiles_per_col;

    int global_tile_idx = blockIdx.x;

    // Índice de la imagen que corresponde a este bloque
    int l = global_tile_idx / tiles_per_image;
    if (l >= L)
        return;

    // Tile dentro de la imagen
    int tile_idx = global_tile_idx % tiles_per_image;
    int tile_row = tile_idx / tiles_per_row;
    int tile_col = tile_idx % tiles_per_row;

    // Índice local del hilo
    int local_row = threadIdx.x / TILE_WIDTH;
    int local_col = threadIdx.x % TILE_WIDTH;

    // Posición global del píxel
    int global_row = tile_row * TILE_WIDTH + local_row;
    int global_col = tile_col * TILE_WIDTH + local_col;

    // Posición dentro del tile con halo
    int tile_row_s = local_row + 1;
    int tile_col_s = local_col + 1;

    // Cargar kernels en memoria compartida
    if (threadIdx.x < 9)
    {
        kernel1_shared[threadIdx.x] = kernel_1[threadIdx.x];
        kernel2_shared[threadIdx.x] = kernel_2[threadIdx.x];
    }
    __syncthreads();

    int img_offset = l * M * N;

    // Cargar centro del tile
    if (global_row < M && global_col < N)
        tile[tile_row_s][tile_col_s] = imgs[img_offset + global_row * N + global_col];
    else
        tile[tile_row_s][tile_col_s] = 0.0f;

    // Cargar halo
    // Laterales
    if (local_col == 0 && global_col > 0)
        tile[tile_row_s][tile_col_s - 1] = imgs[img_offset + global_row * N + (global_col - 1)];
    if (local_col == TILE_WIDTH - 1 && global_col < N - 1)
        tile[tile_row_s][tile_col_s + 1] = imgs[img_offset + global_row * N + (global_col + 1)];

    // Verticales
    if (local_row == 0 && global_row > 0)
        tile[tile_row_s - 1][tile_col_s] = imgs[img_offset + (global_row - 1) * N + global_col];
    if (local_row == TILE_WIDTH - 1 && global_row < M - 1)
        tile[tile_row_s + 1][tile_col_s] = imgs[img_offset + (global_row + 1) * N + global_col];

    // Esquinas
    if (local_row == 0 && local_col == 0 && global_row > 0 && global_col > 0)
        tile[tile_row_s - 1][tile_col_s - 1] = imgs[img_offset + (global_row - 1) * N + (global_col - 1)];
    if (local_row == 0 && local_col == TILE_WIDTH - 1 && global_row > 0 && global_col < N - 1)
        tile[tile_row_s - 1][tile_col_s + 1] = imgs[img_offset + (global_row - 1) * N + (global_col + 1)];
    if (local_row == TILE_WIDTH - 1 && local_col == 0 && global_row < M - 1 && global_col > 0)
        tile[tile_row_s + 1][tile_col_s - 1] = imgs[img_offset + (global_row + 1) * N + (global_col - 1)];
    if (local_row == TILE_WIDTH - 1 && local_col == TILE_WIDTH - 1 &&
        global_row < M - 1 && global_col < N - 1)
        tile[tile_row_s + 1][tile_col_s + 1] = imgs[img_offset + (global_row + 1) * N + (global_col + 1)];

    __syncthreads();

    // Aplicar filtro si no es borde
    if (global_row > 0 && global_row < M - 1 && global_col > 0 && global_col < N - 1)
    {
        float temp1 = 0.0f;
        float temp2 = 0.0f;
        for (int k1 = -1; k1 <= 1; ++k1)
        {
            for (int k2 = -1; k2 <= 1; ++k2)
            {
                int kidx = (k1 + 1) * 3 + (k2 + 1);
                temp1 += tile[tile_row_s + k1][tile_col_s + k2] * kernel1_shared[kidx];
                temp2 += tile[tile_row_s + k1][tile_col_s + k2] * kernel2_shared[kidx];
            }
        }

        float grad = sqrtf(temp1 * temp1 + temp2 * temp2);
        grad = grad > 255.0f ? 255.0f : grad;

        int out_offset = l * (M - 2) * (N - 2);
        out[out_offset + (global_row - 1) * (N - 2) + (global_col - 1)] = grad;
    }
}

/*
 *  Codigo Principal
 */
int main(int argc, char **argv)
{
    clock_t t1, t2;
    double ms;
    cudaEvent_t ct1, ct2;
    float dt;
    int L, M, N, P;
    float *imgs;

    ReadBIN(&imgs, &L, &M, &N, &P, "video_frames_dims.txt", "video_frames.bin");

    float *out = new float[(L * (M - 2) * (N - 2))];

    // CPU
    t1 = clock();
    funcionCPU(imgs, out, L, M, N);
    t2 = clock();
    ms = 1000.0 * (double)(t2 - t1) / CLOCKS_PER_SEC;
    printf("Tiempo CPU: %f[ms]\n", ms);

    WriteBIN(out, L, M - 2, N - 2, "video_out_dims_cpu.txt", "video_out_cpu.bin");

    // GPU
    float *imgsDev, *OutDev, *kernel_1Dev, *kernel_2Dev;
    int block_size = 256;
    int grid_size = (int)ceil((float)M * N / block_size);


    // GPU naive (corregido)
    int total_threads = L * M * N;
    grid_size = (total_threads + block_size - 1) / block_size;

    cudaMalloc(&imgsDev, sizeof(float) * L * M * N);
    cudaMalloc(&OutDev, sizeof(float) * L * (M - 2) * (N - 2));
    cudaMalloc(&kernel_1Dev, sizeof(float) * 9);
    cudaMalloc(&kernel_2Dev, sizeof(float) * 9);

    cudaMemcpy(imgsDev, imgs, sizeof(float) * L * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_1Dev, kernel_1, sizeof(float) * 9, cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_2Dev, kernel_2, sizeof(float) * 9, cudaMemcpyHostToDevice);

    cudaEventCreate(&ct1);
    cudaEventCreate(&ct2);

    cudaEventRecord(ct1);
    kernelNaive<<<grid_size, block_size>>>(imgsDev, OutDev, kernel_1Dev, kernel_2Dev, L, M, N);
    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dt, ct1, ct2);
    cudaMemcpy(out, OutDev, sizeof(float) * L * (M - 2) * (N - 2), cudaMemcpyDeviceToHost);
    printf("Tiempo GPU: %f[ms]\n", dt);

    WriteBIN(out, L, M - 2, N - 2, "video_out_dims_gpu.txt", "video_out_gpu.bin");

    // GPU optimizado
    const int TILE_WIDTH = 16;

    int tiles_per_row = (N + TILE_WIDTH - 1) / TILE_WIDTH;
    int tiles_per_col = (M + TILE_WIDTH - 1) / TILE_WIDTH;
    int tiles_per_image = tiles_per_row * tiles_per_col;

    grid_size = L * tiles_per_image;
    block_size = TILE_WIDTH * TILE_WIDTH;

    cudaEventRecord(ct1);
    kernelOpt<<<grid_size, block_size>>>(imgsDev, OutDev, kernel_1Dev, kernel_2Dev, L, M, N);
    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dt, ct1, ct2);
    cudaMemcpy(out, OutDev, sizeof(float) * L * (M - 2) * (N - 2), cudaMemcpyDeviceToHost);
    printf("Tiempo GPU Opt: %f[ms]\n", dt);

    WriteBIN(out, L, M - 2, N - 2, "video_out_dims_opt.txt", "video_out_opt.bin");

    // Cleanup
    cudaFree(imgsDev);
    cudaFree(OutDev);
    cudaFree(kernel_1Dev);
    cudaFree(kernel_2Dev);
    delete[] imgs;
    delete[] out;

    return 0;
}