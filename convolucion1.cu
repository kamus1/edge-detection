#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

float kernel_1[9] = {1.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0};
float kernel_2[9] = { -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0};

__global__ void kernelOpt(float *img, float *out, float *kernel_1, float *kernel_2, int L, int M, int N)
{
	// Memoria compartida: 16×16 + halo (18×18)   ────────────────
	__shared__ float tile[TILE_H][TILE_W];

	/* -----------------------------------------------------------
	   1. Mapear grid 1-D → coordenadas 2-D de bloque               */
	int blocks_per_row = (N + BLOCK_W - 1) / BLOCK_W;
	int block_row = blockIdx.x / blocks_per_row; // y-block
	int block_col = blockIdx.x % blocks_per_row; // x-block

	/* 2. Coordenadas locales y globales del hilo                  */
	int t_row = threadIdx.x / BLOCK_W; // 0-15 dentro del bloque
	int t_col = threadIdx.x % BLOCK_W;
	int g_row = block_row * BLOCK_H + t_row;
	int g_col = block_col * BLOCK_W + t_col;

	/* -----------------------------------------------------------
	   3. Cargar cooperativamente TODA la sub-matriz 18×18
		  (incluido halo) a shared memory                         */
	for (int idx = threadIdx.x; idx < TILE_H * TILE_W; idx += blockDim.x)
	{
		int r = idx / TILE_W;					   // fila en tile
		int c = idx % TILE_W;					   // col  "
		int img_r = block_row * BLOCK_H + (r - 1); // -1 = halo
		int img_c = block_col * BLOCK_W + (c - 1);

		if (img_r >= 0 && img_r < M && img_c >= 0 && img_c < N)
			tile[r][c] = img[img_r * N + img_c];
		else
			tile[r][c] = 0.0f; // padding cero fuera de imagen
	}
	__syncthreads();

	/* -----------------------------------------------------------
	   4. Convolución 3×3 (solo hilos cuya salida es válida)       */
	if (g_row > 0 && g_row < M - 1 && g_col > 0 && g_col < N - 1)
	{
		float gx = 0.f, gy = 0.f;

		// #pragma unroll
		for (int k1 = -1; k1 <= 1; ++k1)
		{
			// #pragma unroll
			for (int k2 = -1; k2 <= 1; ++k2)
			{
				float pix = tile[t_row + 1 + k1][t_col + 1 + k2];
				int kidx = (k1 + 1) * 3 + (k2 + 1);
				gx += pix * kernel_1[kidx];
				gy += pix * kernel_2[kidx];
			}
		}

		float grad = sqrtf(gx * gx + gy * gy);
		grad = grad > 255.f ? 255.f : grad;
		out[(g_row - 1) * (N - 2) + (g_col - 1)] = grad;
	}
}



//OK
void Read(float** img, int *L, int *M, int *N, int* P, const char *filename) {    
	FILE *fp;
	fp = fopen(filename, "r");
	fscanf(fp, "%d %d %d %d\n", L ,M, N, P); //ahora tambien leemos L

	//guardar todas las imagenes
	int imsize_total = (*L) * (*M) * (*N);
	int imsize = (*M) * (*N); //tamaño original de 1 imagen

	float* img_new = new float[imsize_total];

	for( int j=0; j < (*L); ++j){ //recorrer cantidad de imagenes

		//por cada bloque de imagenes asignar los indices
		for(int i = 0; i < imsize; i++)
			fscanf(fp, "%f ", &(img_new[i + j*imsize]));
	} 
	fclose(fp);
	*img = img_new;
}

void Write(float* imgs, int L, int M, int N, const char *filename) {
	//no es neceario modificar nada, se asume R,G,B tamaño M*N con valores promediados
    FILE *fp;
    fp = fopen(filename, "w");
    fprintf(fp, "%d %d %d\n", L, M, N);
    for(int i = 0; i < L*M*N; i++)
        fprintf(fp, "%f ", imgs[i]);
    fclose(fp);
}

void funcionCPU(float *img, float *out, int L, int M, int N){

	for (int l = 0; l < L; l++) // L fotos
	{
		for (int i = 0; i < M - 2; i++) // M Filas
		{
			for (int j = 0; j < N - 2; j++) // N columnas
			{
				float temp1 = 0;
				float temp2 = 0;

				for (int k1 = 0; k1 < 3; k1++)// kernel
				{
					for (int k2 = 0; k2 < 3; k2++) //kernel
					{
						temp1 += img[(i+k1)*N + (j+k2)] * kernel_1[k1*3 + k2];
						temp2 += img[(i+k1)*N + (j+k2)] * kernel_2[k1*3 + k2];
					}
				}			
				out[i*N + j] = sqrt(pow(temp1, 2) + pow(temp2, 2)) > 255.0 ? 255:sqrt(pow(temp1, 2) + pow(temp2, 2));
			}
		}
	}
}

__global__ void kernelNaive(float *imgs, float *out, float *kernel_1, float *kernel_2, int L, int M, int N) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int imsize = M * N;

    if (tid < imsize) { 
		float temp1 = 0;
		float temp2 = 0;
		int i = tid / N;
		int j = tid % N; 

		for (int k1 = 0; k1 < 3; k1++)// kernel
		{
			for (int k2 = 0; k2 < 3; k2++) //kernel
			{
				temp1 += imgs[(i+k1)*N + (j+k2)] * kernel_1[k1*3 + k2];
				temp2 += imgs[(i+k1)*N + (j+k2)] * kernel_2[k1*3 + k2];
			}
		}			
		out[i*N + j] = sqrt(temp1*temp1 + temp2*temp2) > 255.0 ? 255: sqrt(temp1*temp1 + temp2*temp2);
    }
}



/*
 *  Codigo Principal
 */
int main(int argc, char **argv){

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

	float *out = (float*) malloc(sizeof(float)*N*M*L);
    /*
     *  Parte CPU
     */
    t1 = clock();
	funcionCPU(imgs,  out, L, M, N);
	t2 = clock();
	ms = 1000.0 * (double)(t2 - t1) / CLOCKS_PER_SEC;
	printf("Tiempo CPU: %f[ms]\n", ms);

	Write(out, L, M, N, "out.txt");
	/*
	 *  Parte GPU
	 */
	float *imgsDev;
	float *OutDev;
	float *kernel_1Dev;
	float *kernel_2Dev;
	int block_size = 256;
	int grid_size = (int) ceil((float) M * N / block_size );

  cudaMalloc((void**)&imgsDev, sizeof(float)*L*M*N);
	cudaMalloc((void**)&OutDev, sizeof(float)*L*M*N);
	cudaMalloc((void**)&kernel_1Dev, sizeof(float)*9);
	cudaMalloc((void**)&kernel_2Dev, sizeof(float)*9);

  cudaMemcpy(imgsDev, imgs, L*M*N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(kernel_1Dev, kernel_1, 9 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(kernel_2Dev, kernel_2, 9 * sizeof(float), cudaMemcpyHostToDevice);
	    
  cudaEventCreate(&ct1);
  cudaEventCreate(&ct2);
  cudaEventRecord(ct1);
  kernelNaive<<<grid_size, block_size>>>(imgsDev, OutDev, kernel_1Dev, kernel_2Dev, L, M, N);
  cudaEventRecord(ct2);
  cudaEventSynchronize(ct2);
  cudaEventElapsedTime(&dt, ct1, ct2);
  cudaMemcpy(out, OutDev, L*M*N*sizeof(float), cudaMemcpyDeviceToHost);
	printf("Tiempo GPU: %f[ms]\n", dt);
	Write(out, L, M, N, "out_dev.txt");

	cudaFree(imgsDev); cudaFree(OutDev); cudaFree(kernel_1Dev); cudaFree(kernel_2Dev);
	
	delete[] imgs; delete[] out;

	return 0;
}