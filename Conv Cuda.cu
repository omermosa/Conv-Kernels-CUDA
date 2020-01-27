// Conv. operations on images using CUDA
// Copyrights Omer Hassan, All Rights Reserved
// The American University in Cairo (AUC)

//The images are loaded and processed using cimg.h library

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#ifndef cimg_debug
#define cimg_debug 1
#endif
#include "CImg.h"
#include<time.h>
using namespace cimg_library;
using namespace std;
#define TW 8
#define MASK_WIDTH 3
#define shared_X (TW +MASK_WIDTH - 1)
#define shared_y (TW + MASK_WIDTH - 1)
#define lp(i,n,s) for(int i=s;i<n;i++) 
__constant__ float M[3][3];

__shared__ float N_ds[TW][TW];
//global Var
float kt, kernel_time, parallel_time;

__global__ void ConvolutionKernel(unsigned char* img, unsigned char* outimg, int w, int h)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int row = by * TW + ty;
	int col = bx * TW + tx;
	if (row < h && col < w && row >= 0 && col >= 0) {
		N_ds[ty][tx] = img[row * w + col];

		__syncthreads();

		int row_start = row - MASK_WIDTH / 2;
		int col_start = col - MASK_WIDTH / 2;
		float sum = 0;
		int Cblk_col = bx * TW;
		int Nblk_col = (bx + 1) * TW;
		int Cblk_row = by * TW;
		int Nblk_row = (by + 1) * TW;
		//Caching
		if (ty < TW && tx < TW) {
			for (int i = 0; i < MASK_WIDTH; i++) {
				for (int j = 0; j < MASK_WIDTH; j++) {
					int row_index = row_start + i;
					int col_index = col_start + j;
					if (row_index >= 0 && row_index < h && col_index >= 0 && col_index < w) {
						if ((row_index >= Cblk_row) && (row_index < Nblk_row) && (col_index >= Cblk_col) && (col_index < Nblk_col))
							sum += N_ds[ty + i - (MASK_WIDTH / 2)][tx + j - (MASK_WIDTH / 2)] * M[i][j];//element in shared mem
						else sum += M[i][j] * img[w * row_index + col_index]; //rely on caching
					}
					else { //replicating values
						//row and cols are out of img
						if (row_index < 0 && col_index < 0)sum += M[i][j] * img[0];
						else if (row_index < 0 && col_index >= w)sum += M[i][j] * img[w - (MASK_WIDTH / 2)];
						else if (row_index >= h && col_index < 0)sum += M[i][j] * img[w * (h - MASK_WIDTH / 2)];
						else if (row_index >= h && col_index >= w)sum += M[i][j] * img[w * (h - MASK_WIDTH / 2) + w - (MASK_WIDTH / 2)];

						else if (row_index >= 0 && row_index < h) //row is fine, check col
						{
							if (col_index < 0) sum += M[i][j] * img[w * row_index + col_index + (MASK_WIDTH / 2)];
							else if (col_index >= w) sum += M[i][j] * img[w * row_index + col_index - (MASK_WIDTH / 2)];
						}
						else if (col_index >= 0 && col_index < w) //col is fine, check row
						{
							if (row_index < 0) sum += M[i][j] * img[w * (row_index + (MASK_WIDTH / 2)) + col_index];
							else if (row_index >= h) sum += M[i][j] * img[w * (row_index - (MASK_WIDTH / 2)) + col_index];
						}}

					
				}

			}
		
			if (sum < 0)sum = 0; //clipping zeros
			if (sum > 255)sum = 255; //clipping bigger than 255
			outimg[row * w + col] = sum;
		}
	}
}
void ConvWrapper(unsigned char** img, int w, int h, unsigned char** outimg,  float Mask[][3])
{
	unsigned char* img1D = new unsigned char[w * h];
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			img1D[i * w + j] = img[i][j];

		}

	}
	unsigned char* outimg1D = new unsigned char[w * h];
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			outimg1D[i * w + j] = outimg[i][j];
		}

	}
	double s = clock(); //start time

	int size = w * h * sizeof(unsigned char);
	unsigned char* d_img, * d_outimg;

	cudaError_t err = cudaMalloc((void**)& d_img, size);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(d_img, img1D, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void**)& d_outimg, size);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	cudaEvent_t start, stop;
	float elapsedTime;

	//cudaEventCreate(&start);
	//cudaEventRecord(start, 0);
	double ss = clock();
	dim3 dimBlock(TW, TW, 1);
	dim3 dimGrid(ceil(w / float(TW)), ceil(h/ float(TW)), 1);


	cudaMemcpyToSymbol(M, Mask, MASK_WIDTH* MASK_WIDTH *sizeof(float));
	//Kernek Call
	//ConvolutionKernel << <dimGrid, dimBlock >> > (d_img, d_outimg, w,h);
	ConvolutionKernel << <dimGrid, dimBlock >> > (d_img, d_outimg, w, h);
	double ee = clock();
	kt = (double)(ee - ss) / CLOCKS_PER_SEC;
	//cudaEventCreate(&stop);
	//cudaEventRecord(stop, 0);
	//cudaEventSynchronize(stop);

	//cudaEventElapsedTime(&kernel_time, start, stop);
	//printf("Elapsed time by the Kernel: %f ms\n", kernel_time);
	printf("Elapsed time by the kernel: %f \n", kt);
	err = cudaMemcpy(outimg1D, d_outimg, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	//end time
	double e = clock();
	parallel_time= (double)(e - s) / CLOCKS_PER_SEC;


	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			outimg[i][j] = outimg1D[i * w + j];

		}
	}
	cudaFree(d_img); cudaFree(d_outimg); 
	free(img1D); free(outimg1D);
}

bool compimgs(unsigned char** seq_img, unsigned char** par_img, int w, int h) {
	lp(i, h, 1) lp(j, w, 1) if (abs(seq_img[i][j] - par_img[i][j]) > 0.0001) {
		cout << i << " " << j << " " << int(seq_img[i][j]) << " " << int(par_img[i][j]) << endl; return false;
	}
	return true;


}

int main()
{
	// Masks 
	float blur[3][3] = { 0.0625, 0.125, 0.0625,
		0.125, 0.25, 0.125,
		0.0625, 0.125, 0.0625 };
	float emboss[3][3] = { -2, -1, 0,
		-1 ,1 ,1,
		0 ,1 ,2 };
	float outline[3][3] = { -1 , -1 ,-1
		- 1 ,  8 ,-1
		- 1 , -1 ,-1 };

	float sharpen[3][3] = { 0.0,-1.0,0.0,-1.0,5.0,-1.0,0.0,-1.0,0.0 };
	float left_sobel[3][3] = { 1,0,-1,2,0,-2,1,0,-1 };
	float right_sobel[3][3] = { -1,0,1,-2,0,2,-1,0,1 };
	float top_sobel[3][3] = { 1,2,1,0,0,0,-1,-2,-1 };
	float bottom_sobel[3][3] = { -1,-2,-1,0,0,0,1,2,1 };
	float identity[3][3] = { 0,0,0,0,1,0,0,0,0 };
	//lp(i, 3, 0) lp(j, 3, 0) cout << sharpen[i][j] << " ";
// load the image
	char imgpath[100]; string imgname;
	int op;
	float mask[MASK_WIDTH][MASK_WIDTH];

	cout << "Enter image directory " << endl;
	cin >> imgpath;
	cout << "Enter the operation number required: 1:blur, 2: emboss, 3: outline, 4: sharpen, 5: left_sobel, 6: right_sobel, 7: top_sobel, 8: bottom_sobel " << endl;
	cin >> op;
	if (op == 1)memcpy(mask,blur,MASK_WIDTH*MASK_WIDTH*sizeof(float));
	else if(op==2)memcpy(mask, emboss, MASK_WIDTH * MASK_WIDTH * sizeof(float));
	else if (op == 3)memcpy(mask, outline, MASK_WIDTH * MASK_WIDTH * sizeof(float));
	else if (op == 4)memcpy(mask, sharpen, MASK_WIDTH * MASK_WIDTH * sizeof(float));
	else if (op == 5)memcpy(mask, left_sobel, MASK_WIDTH * MASK_WIDTH * sizeof(float));
	else if (op == 6)memcpy(mask, right_sobel, MASK_WIDTH * MASK_WIDTH * sizeof(float));
	else if (op == 7)memcpy(mask, top_sobel, MASK_WIDTH * MASK_WIDTH * sizeof(float));
	else if (op == 8)memcpy(mask, bottom_sobel, MASK_WIDTH * MASK_WIDTH * sizeof(float));
	else memcpy(mask, identity, MASK_WIDTH * MASK_WIDTH * sizeof(float));

	CImg< unsigned char> image((imgpath));
	int w = image.width();
	int h = image.height();
	image.channel(0);
	
	CImg<float> outimg_seq(w, h, 1, 1, 0);
	CImg<float> outimg_par(w, h, 1, 1, 0);
	CImg<float> testimg(w, h, 1, 1, 0);

	int wp = w + MASK_WIDTH -1;
	int hp = h + MASK_WIDTH-1 ;

	unsigned char** out_img = new unsigned char* [h];
	lp(i, h, 0) {
		out_img[i] = new unsigned char[w];
	}
	unsigned char** out_seq = new unsigned char* [h];
	lp(i, h, 0) {
		out_seq[i] = new unsigned char[w];
	}

	unsigned char** img_par=new unsigned char*[h];
	unsigned char** img_seq = new unsigned char* [h];

	lp(i,h,0){
		img_par[i] = new unsigned char[w];
	}
	double s = clock();
	lp(i, h, 0) {
		img_seq[i] = new unsigned char[w];
	}

	lp(i, h,0)
		lp(j, w, 0) {
		img_par[i][j] = (image(j, i, 0, 0));
	}
	lp(i, h, 0)
		lp(j, w, 0) {
		img_seq[i][j] = (image(j, i, 0, 0));
	}
	double e = clock();
	lp(i, h, 0)
		lp(j, w, 0) {
	  testimg(j, i, 0, 0)=img_par[i][j];
	}

	testimg.save("test.jpg");

	// paralel version
	ConvWrapper(img_par, w, h, out_img, mask);
	cout << "Device Time " << parallel_time << endl;
	lp(i, h, 0)
		lp(j, w, 0) {
		outimg_par(j, i, 0, 0) = out_img[i][j];
	}
	float test[10][10];
	lp(i, 10, 0) lp(j, 10, 0) test[i][j] = 10;
	 float testres[8][8];


	 //Sequential Code


//Replicating edges
  //lp(i, wp - 1, 1) img_seq[0][i] = img_seq[1][i];
  // lp(i, wp - 1, 1) img_seq[hp-1][i] = img_seq[hp-2][i];
  // lp(i,hp - 1, 1) img_seq[i][0] = img_seq[i][1];
  // lp(i, hp - 1, 1) img_seq[i][wp-1] = img_seq[i][wp-2];
  // img_seq[0][0] = img_seq[1][1];
  // img_seq[hp-1][0] = img_seq[hp-2][1];
  // img_seq[0][wp-1] = img_seq[1][wp-2];
  // img_seq[hp - 1][wp-1] = img_seq[hp - 2][wp-2];
   /*lp(i, h) {
	   lp(j, w) cout << int (img[i][j]) << " ";
	   cout << endl;
   }*/
   long long int countop = 0;

	double start = clock();
	float sum;
	for (int i =0; i < h ; i++) {
		for (int j = 0; j <w; j++) {
			sum = 0.0f;

			for (int y = -1; y <= 1; y++)
				for (int x = -1; x <= 1; x++)
				{
					if (i + y >= 0 && i + y < h && j + x >= 0 && j + x < w) {
						sum += img_seq[i + y][j + x] * mask[y + 1][x + 1];
					}
	
						else { //replicating values
							//row and cols are out of img
							if (i+y < 0 && j+x < 0)sum += mask[y + 1][x + 1] * img_seq[0][0];
							else if (i+y < 0 && j+x >= w)sum += mask[y + 1][x + 1] * img_seq[0][w - (MASK_WIDTH / 2)];
							else if (i+y >= h && j+x < 0)sum += mask[y + 1][x + 1] * img_seq[(h - MASK_WIDTH / 2)][0];
							else if (i+y >= h && j+x >= w)sum += mask[y + 1][x + 1] * img_seq[(h - MASK_WIDTH / 2)][w - (MASK_WIDTH / 2)];

							else if (i+y >= 0 && i+y < h) //row is fine, check col
							{
								if (j+x < 0) sum += mask[y+1][x+1] * img_seq[i+y][j+x + (MASK_WIDTH / 2)];
								else if (j+x >= w) sum += mask[y+1][x+1] * img_seq[i+y][j+x - (MASK_WIDTH / 2)];
							}
							else if (j+x >= 0 && j+x < w) //col is fine, check row
							{
								if (i+y < 0) sum += mask[y + 1][x + 1] * img_seq[i+y + (MASK_WIDTH / 2)][j+x];
								else if (i+y >= h) sum += mask[y + 1][x + 1] * img_seq[i+y - (MASK_WIDTH / 2)][j+x];
							}
						}
					
						countop+=2; //every iteration, 2 float operations 
					}
			if (sum < 0)sum = 0; //clipping zeros
			if (sum > 255)sum = 255; //clipping bigger than 255
			out_seq[i][j] = sum;
		}
	}
	lp(i, h, 0)
		lp(j, w, 0) {
		outimg_seq(j, i, 0, 0) = out_seq[i][j];
	}
	
	if (compimgs(out_seq, out_img, w, h)) cout << "identical pixel values- the images are identical "<<endl;
	else cout << "different pixel values " << endl;
	double end = clock();
	double seq_time = double((end - start)+(e-s)) / CLOCKS_PER_SEC;
	cout << "Host Time " << seq_time << endl;
	cout << "Speed UP Kernel Only- No Overhead " << seq_time / kt << endl;

	cout << "Speed UP -Overhead " << seq_time / (parallel_time) << endl;

	cout << "Sequential GFLOPS " << countop / float(1000000000) / seq_time << endl;

	cout << "Parallel GFLOPS " << countop / float(1000000000) / (kt);
	cout << endl;
	
outimg_seq.save("out_seq.jpg");

outimg_par.save("out_par.jpg");


	return 0;
}
