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
void Read(float **img, int *L, int *M, int *N, int *P, const char *filename)
{
	FILE *fp;
	fp = fopen(filename, "r");
	fscanf(fp, "%d %d %d %d\n", L, M, N, P); // ahora tambien leemos L

	// guardar todas las imagenes
	int imsize = (*M) * (*N); // tamaño original de 1 imagen
	int imsize_total = (*L) * imsize;

	float *img_new = new float[imsize_total];

	for (int j = 0; j < (*L); ++j)
	{ // recorrer cantidad de imagenes

		// por cada bloque de imagenes asignar los indices
		for (int i = 0; i < imsize; i++)
			fscanf(fp, "%f ", &(img_new[i + j * imsize]));
	}
	fclose(fp);
	*img = img_new;
}

void Write(float *imgs, int L, int M, int N, const char *filename)
{
	// no es neceario modificar nada, se asume R,G,B tamaño M*N con valores promediados
	FILE *fp;
	fp = fopen(filename, "w");
	fprintf(fp, "%d %d %d\n", L, M, N);
	for (int i = 0; i < L * M * N; i++)
		fprintf(fp, "%f ", imgs[i]);
	fclose(fp);
}

void funcionCPU(float *img, float *out, int L, int M, int N)
{

	for (int l = 0; l < L; l++) // L fotos
	{
		for (int i = 1; i < M - 1; i++) // M Filas
		{
			for (int j = 1; j < N - 1; j++) // N columnas
			{
				float temp1 = 0;
				float temp2 = 0;

				for (int k1 = -1; k1 < 2; k1++) // kernel
				{
					for (int k2 = -1; k2 < 2; k2++) // kernel
					{
						temp1 += img[(l * N * M) + (i + k1) * (N) + (j + k2)] * kernel_1[(k1 + 1) * 3 + (k2 + 1)];
						temp2 += img[(l * N * M) + (i + k1) * (N) + (j + k2)] * kernel_2[(k1 + 1) * 3 + (k2 + 1)];
					}
				}
				out[(l * (N - 2) * (M - 2)) + (i - 1) * (N - 2) + (j - 1)] = sqrt(temp1 * temp1 + temp2 * temp2) > 255.0 ? 255.0 : sqrt(temp1 * temp1 + temp2 * temp2);
			}
		}
	}
}

__global__ void kernelNaive(float *imgs, float *out, float *kernel_1, float *kernel_2, int L, int M, int N)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int imsize = M * N;

	if (tid < imsize)
	{
		float temp1 = 0;
		float temp2 = 0;
		int i = tid / N;
		int j = tid % N;

		if (i > 0 && i < M - 1 && j > 0 && j < N - 1)
		{
			for (int k1 = -1; k1 < 2; k1++) // kernel
			{
				for (int k2 = -1; k2 < 2; k2++) // kernel
				{
					temp1 += imgs[(i + k1) * N + (j + k2)] * kernel_1[(k1 + 1) * 3 + (k2 + 1)];
					temp2 += imgs[(i + k1) * N + (j + k2)] * kernel_2[(k1 + 1) * 3 + (k2 + 1)];
				}
			}
			out[(i - 1) * (N - 2) + (j - 1)] = sqrt(temp1 * temp1 + temp2 * temp2) > 255.0 ? 255.0 : sqrt(temp1 * temp1 + temp2 * temp2);
		}
	}
}

__global__ void kernelOpt(float *img, float *out, float *kernel_1, float *kernel_2, int M, int N) {
    const int TILE_WIDTH = 16;
    //const int TILE_PIXELS = TILE_WIDTH * TILE_WIDTH;
    const int SHARED_WIDTH = TILE_WIDTH + 2;

    __shared__ float tile[SHARED_WIDTH][SHARED_WIDTH];
		__shared__ float kernel1_shared[9];
		__shared__ float kernel2_shared[9];

    // Índice del hilo global
    //int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Número de tiles por fila
    int tiles_per_row = (N + TILE_WIDTH - 1) / TILE_WIDTH;

    // Identificador del bloque como si fuera una grilla 2D
    int tile_row = blockIdx.x / tiles_per_row;
    int tile_col = blockIdx.x % tiles_per_row;

    // Índice local del hilo en el bloque
    int local_row = threadIdx.x / TILE_WIDTH;
    int local_col = threadIdx.x % TILE_WIDTH;

    // Posición global del píxel en la imagen
    int global_row = tile_row * TILE_WIDTH + local_row;
    int global_col = tile_col * TILE_WIDTH + local_col;

    // Posición en tile con halo
    int tile_row_s = local_row + 1;
    int tile_col_s = local_col + 1;

	if (threadIdx.x < 9) {            // <- usa threadIdx.x
		kernel1_shared[threadIdx.x] = kernel_1[threadIdx.x];
		kernel2_shared[threadIdx.x] = kernel_2[threadIdx.x];
	}
	__syncthreads();    
		  
    // Cargar el centro del tile
    if (global_row < M && global_col < N)
        tile[tile_row_s][tile_col_s] = img[global_row * N + global_col];
    else
        tile[tile_row_s][tile_col_s] = 0.0f;

    // Cargar halo
    // Izquierda y derecha
    if (local_col == 0 && global_col > 0)
        tile[tile_row_s][tile_col_s - 1] = img[global_row * N + (global_col - 1)];
    if (local_col == TILE_WIDTH - 1 && global_col < N - 1)
        tile[tile_row_s][tile_col_s + 1] = img[global_row * N + (global_col + 1)];

    // Arriba y abajo
    if (local_row == 0 && global_row > 0)
        tile[tile_row_s - 1][tile_col_s] = img[(global_row - 1) * N + global_col];
    if (local_row == TILE_WIDTH - 1 && global_row < M - 1)
        tile[tile_row_s + 1][tile_col_s] = img[(global_row + 1) * N + global_col];

    // Esquinas
    if (local_row == 0 && local_col == 0 && global_row > 0 && global_col > 0)
        tile[tile_row_s - 1][tile_col_s - 1] = img[(global_row - 1) * N + (global_col - 1)];
    if (local_row == 0 && local_col == TILE_WIDTH - 1 && global_row > 0 && global_col < N - 1)
        tile[tile_row_s - 1][tile_col_s + 1] = img[(global_row - 1) * N + (global_col + 1)];
    if (local_row == TILE_WIDTH - 1 && local_col == 0 && global_row < M - 1 && global_col > 0)
        tile[tile_row_s + 1][tile_col_s - 1] = img[(global_row + 1) * N + (global_col - 1)];
    if (local_row == TILE_WIDTH - 1 && local_col == TILE_WIDTH - 1 &&
        global_row < M - 1 && global_col < N - 1)
        tile[tile_row_s + 1][tile_col_s + 1] = img[(global_row + 1) * N + (global_col + 1)];

    __syncthreads();

    // Aplicar convolución solo si no estamos en el borde de la imagen
    if (global_row > 0 && global_row < M - 1 && global_col > 0 && global_col < N - 1) {
        float temp1 = 0.0f;
        float temp2 = 0.0f;
        for (int k1 = -1; k1 <= 1; ++k1) {
            for (int k2 = -1; k2 <= 1; ++k2) {
                int kidx = (k1 + 1) * 3 + (k2 + 1);
                temp1 += tile[tile_row_s + k1][tile_col_s + k2] * kernel1_shared[kidx];
                temp2 += tile[tile_row_s + k1][tile_col_s + k2] * kernel2_shared[kidx];
            }
        }

        float grad = sqrtf(temp1 * temp1 + temp2 * temp2);
        out[(global_row - 1) * (N - 2) + (global_col - 1)] = grad > 255.0f ? 255.0f : grad;
    }
}





/*
 *  Codigo Principal
 */
int main(int argc, char **argv)
{

	/*
	 *  Inicializacion
	 */
	clock_t t1, t2;
	double ms;
	cudaEvent_t ct1, ct2;
	float dt;
	int L, M, N, P;
	float *imgs;
	Read(&imgs, &L, &M, &N, &P, "imagen2.txt");

	float *out = (float *)malloc(sizeof(float) * (N - 2) * (M - 2) * L);
	/*
	 *  Parte CPU
	 */
	t1 = clock();
	funcionCPU(imgs, out, L, M, N);
	t2 = clock();
	ms = 1000.0 * (double)(t2 - t1) / CLOCKS_PER_SEC;
	printf("Tiempo CPU: %f[ms]\n", ms);

	Write(out, L, (M - 2), (N - 2), "out.txt");
	/*
	 *  Parte GPU
	 */
	float *imgsDev;
	float *OutDev;
	float *kernel_1Dev;
	float *kernel_2Dev;
	int block_size = 256;
	int grid_size = (int)ceil((float)M * N / block_size);

	cudaMalloc((void **)&imgsDev, sizeof(float) * L * M * N);
	cudaMalloc((void **)&OutDev, sizeof(float) * L * (M - 2) * (N - 2));
	cudaMalloc((void **)&kernel_1Dev, sizeof(float) * 9);
	cudaMalloc((void **)&kernel_2Dev, sizeof(float) * 9);

	cudaMemcpy(imgsDev, imgs, L * M * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(kernel_1Dev, kernel_1, 9 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(kernel_2Dev, kernel_2, 9 * sizeof(float), cudaMemcpyHostToDevice);

	cudaEventCreate(&ct1);
	cudaEventCreate(&ct2);
	cudaEventRecord(ct1);
	kernelNaive<<<grid_size, block_size>>>(imgsDev, OutDev, kernel_1Dev, kernel_2Dev, L, M, N);
	cudaEventRecord(ct2);
	cudaEventSynchronize(ct2);
	cudaEventElapsedTime(&dt, ct1, ct2);
	cudaMemcpy(out, OutDev, L * (M - 2) * (N - 2) * sizeof(float), cudaMemcpyDeviceToHost);
	printf("Tiempo GPU: %f[ms]\n", dt);
	Write(out, L, (M - 2), (N - 2), "out_dev.txt");

	/*Versión shared mem */

	cudaEventRecord(ct1);
	//int grid_size2 = (int)ceil((float)(M * N) / (block_size * 16));
	kernelOpt<<<grid_size, block_size>>>(imgsDev, OutDev, kernel_1Dev, kernel_2Dev, M, N);
	cudaEventRecord(ct2);
	cudaEventSynchronize(ct2);
	cudaEventElapsedTime(&dt, ct1, ct2);
	cudaMemcpy(out, OutDev, L * (M - 2) * (N - 2) * sizeof(float), cudaMemcpyDeviceToHost);
	printf("Tiempo GPU Opt: %f[ms]\n", dt);
	Write(out, L, (M - 2), (N - 2), "out_dev_opt.txt");

	cudaFree(imgsDev);
	cudaFree(OutDev);
	cudaFree(kernel_1Dev);
	cudaFree(kernel_2Dev);

	delete[] imgs;
	delete[] out;

	return 0;
}