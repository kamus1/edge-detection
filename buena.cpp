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
