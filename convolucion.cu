#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define BLOCK_W 16
#define BLOCK_H 16

float kernel_1[9] = {1.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0};
float kernel_2[9] = { -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0};

//OK
void Read(float** img, int *L, int *M, int *N, int* P, const char *filename) {
	FILE *fp;
	fp = fopen(filename, "r");
	fscanf(fp, "%d %d %d %d\n", L ,M, N, P); //ahora tambien leemos L

	//guardar todas las imagenes
	int imsize = (*M) * (*N); //tamaño original de 1 imagen
	int imsize_total = (*L) * imsize;

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
		for (int i = 1; i < M - 1; i++) // M Filas
		{
			for (int j = 1; j < N - 1 ; j++) // N columnas
			{
				float temp1 = 0;
				float temp2 = 0;

				for (int k1 = -1; k1 < 2; k1++)// kernel
				{
					for (int k2 = -1; k2 < 2; k2++) //kernel
					{
						temp1 += img[(l*N*M) + (i+k1)*(N) + (j+k2)] * kernel_1[(k1 + 1)*3 + (k2 + 1)];
						temp2 += img[(l*N*M) + (i+k1)*(N) + (j+k2)] * kernel_2[(k1+ 1)*3 + (k2 + 1)];
					}
				}
				out[(l*(N-2)*(M-2)) + (i-1)*(N-2) + (j-1)] = sqrt(temp1*temp1 + temp2*temp2) > 255.0 ? 255.0: sqrt(temp1*temp1 + temp2*temp2);
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

    if(i > 0 && i < M-1 && j > 0 && j < N -1 ){
		for (int k1 = -1; k1 < 2; k1++)// kernel
		{
			for (int k2 = -1; k2 < 2; k2++) //kernel
			{
				temp1 += imgs[(i+k1)*N + (j+k2)] * kernel_1[(k1 + 1)*3 + (k2 + 1)];
				temp2 += imgs[(i+k1)*N + (j+k2)] * kernel_2[(k1 +1)*3 + (k2 + 1)];
			}
		}
		out[(i-1)*(N-2) + (j-1)] = sqrt(temp1*temp1 + temp2*temp2) > 255.0 ? 255.0: sqrt(temp1*temp1 + temp2*temp2);
    }
  }
}

__global__ void kernelOpt(float *img, float *out, float *kernel_1, float *kernel_2, int L, int M, int N){
    //Memoria compartida porque se reutilizan pixeles. si tenemos bloques de 256, tenemos que cargar un bloque de 18x18
    __shared__ float tile[18][18];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    //int total_threads = blockDim.x * gridDim.x;
    int num_pixels = M * N;

    // Cada hilo se encarga de 1 píxel
    if (tid < num_pixels) {

    //indice global
    int i = tid / N;  // fila
    int j = tid % N;  // columna


    int tile_col = tid % 16;
    int tile_row = threadIdx.x / 16;

    //cargamos el centro de la imagen con offset 1 para centrar la imagen
    tile[tile_row+1][tile_col+1] = img[i*N + j];
    //falta cargar el halo, no cacho cómo hacerlo

    __syncthreads();

     // Aplicar convolución solo si estamos lejos del borde de la imagen
    if (i > 0 && i < M - 1 && j > 0 && j < N - 1) {
        float temp1 = 0.0f;
        float temp2 = 0.0f;

        for (int k1 = -1; k1 <= 1; k1++) {
            for (int k2 = -1; k2 <= 1; k2++) {
                int idx_row = (tile_row + k1);
                int idx_col = (tile_col + k2);
                int kidx = (k1 + 1) * 3 + (k2 + 1);
                temp1 += tile[idx_row][idx_col] * kernel_1[kidx];
                temp2 += tile[idx_row][idx_col] * kernel_2[kidx];
            }
        }

        float grad = sqrt(temp1 * temp1 + temp2 * temp2);
        out[(i-1) * (N-2) + (j-1)] = grad > 255.0f ? 255.0f : grad;
    }

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

	float *out = (float*) malloc(sizeof(float)*(N-2)*(M-2)*L);
    /*
     *  Parte CPU
     */
  t1 = clock();
	funcionCPU(imgs,  out, L, M, N);
	t2 = clock();
	ms = 1000.0 * (double)(t2 - t1) / CLOCKS_PER_SEC;
	printf("Tiempo CPU: %f[ms]\n", ms);

	Write(out, L, (M-2), (N-2), "out.txt");
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
	cudaMalloc((void**)&OutDev, sizeof(float)*L*(M-2)*(N-2));
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
  cudaMemcpy(out, OutDev, L*(M-2)*(N-2)*sizeof(float), cudaMemcpyDeviceToHost);
	printf("Tiempo GPU: %f[ms]\n", dt);
	Write(out, L, (M-2), (N-2), "out_dev.txt");

  /*Versión shared mem */

  cudaEventRecord(ct1);
  kernelOpt<<<grid_size, block_size>>>(imgsDev, OutDev, kernel_1Dev, kernel_2Dev, L, M, N);
  cudaEventRecord(ct2);
  cudaEventSynchronize(ct2);
  cudaEventElapsedTime(&dt, ct1, ct2);
  cudaMemcpy(out, OutDev, L*(M-2)*(N-2)*sizeof(float), cudaMemcpyDeviceToHost);
	printf("Tiempo GPU Opt: %f[ms]\n", dt);
	Write(out, L, (M-2), (N-2), "out_dev_opt.txt");

	cudaFree(imgsDev); cudaFree(OutDev); cudaFree(kernel_1Dev); cudaFree(kernel_2Dev);

	delete[] imgs; delete[] out;

	return 0;
}