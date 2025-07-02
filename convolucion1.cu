#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

float kernel_1[9] = {1.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0};
float kernel_2[9] = { -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0};

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