# Conv-Kernels-CUDA

This project performs operations on images using Conv Kernels (Masks) and parallelizes these operations using CUDA C. It then compares the timing of both sequential and parallel codes and calculates the speed up. The operations supported by this program are Blurring, Emboss, Outline, Sharpen, Left Sobel, Right Sobel, Top Sobel, Bottom Sobel, Identity. The negative pixel values are all clipped to zeroes.

Reading images as 2D Matrices is done using Cimg.h library (downloaded from their official website). 
