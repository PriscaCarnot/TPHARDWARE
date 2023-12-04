#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <cuda_runtime.h>



// Initialise la matrice entre -1 et 1 
void MatrixInit (float *M, int n, int p, bool a){
 
 for (int i = 0; i< n; i++){
  for (int j = 0; j<p; j++){
   if (a==true){
    float number = rand();
    float randomValue = number / RAND_MAX;
    randomValue = 2*randomValue -1;
   
   //printf("Number %f \n",randomValue);
   
    *(M+i*p+j) = randomValue;
   }
   else {*(M+i*p+j) = 0;}
  }
 }
}

// Affiche la matrice sur le terminal 
void MatrixPrint(float *M, int n, int p){
 for (int i = 0; i< n; i++){
  for (int j = 0; j<p; j++){
   printf("%.2f \t", *(M+i*p+j));
   
  }
  printf("\n");
 }
}

void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
 for (int i = 0; i< n; i++){
  for (int j = 0; j<p; j++){
   float numberM1 = *(M1 +i*p+j);
   float numberM2 = *(M2 +i*p+j);
   float numberOut = numberM1 + numberM2;
   //printf("%f \n", i*p+j);
   *(Mout +i*p+j) = numberOut;
  }
 }
}

/*
// Additionne deux matrices sur le GPU
__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < p) {
        int index = i * p + j;
        Mout[index] = M1[index] + M2[index];
    }
}
*/

__global__ void cudaMatrixAdd(float *M1, float *M2, float *MoutGPU, int n, int p){
  int i = blockIdx.x;
  int j = threadIdx.x;
  

  
  float numberM1 = *(M1 +i*p+j);
  float numberM2 = *(M2 +i*p+j);
  float numberOut = numberM1 + numberM2;
  *(MoutGPU +i*p+j) = numberOut;
  
}


void MatrixMult(float *M1, float *M2, float *Mout, int n){
 for (int i = 0; i< n; i++){
  for (int j = 0; j<n; j++){
   *(Mout +i*n+j) = 0;
   for (int m = 0; m<n; m++){
    float numberM1 = *(M1 +i*n+m);
    float numberM2 = *(M2 +m*n+j);
    float numberOut = numberM1 * numberM2;
    *(Mout +i*n+j) += numberOut;
   }
  }
 }
}


__global__ void cudaMatrixMult(float *M1, float *M2, float *MoutGPU, int n){
  int i = blockIdx.x;
  int j = threadIdx.x;
  *(MoutGPU +i*n+j) = 0;
  for (int m = 0; m<n; m++){
    float numberM1 = *(M1 +i*n+m);
    float numberM2 = *(M2 +m*n+j);
    float numberOut = numberM1 * numberM2;
    *(MoutGPU +i*n+j) += numberOut;
   }
}



int main() {
  int n = 32; // taille raw data
  int p = 28;
  int c = 6; 
  float  raw_data[n][n], C1_data[c][p][p], Mout[n][p], MoutGPU[n][p], MoutGPU2[n][p];
  
  // Avec le CPU : 
  MatrixInit(&raw_data[0][0], n, n, true);
  MatrixPrint(&raw_data[0][0], n, n);

  
  /*
  // Avec le GPU : 
  float *d_M1, *d_M2, *d_Mout, *d_Mout2;

  
  cudaMalloc((void**)&d_M1, (n * p) * sizeof(float));
  cudaMalloc((void**)&d_M2, (n * p) * sizeof(float));
  cudaMalloc((void**)&d_Mout, (n * p) * sizeof(float));
  cudaMalloc((void**)&d_Mout2, (n * p) * sizeof(float));

    
  cudaMemcpy(d_M1, M1, (n * p) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_M2, M2, (n * p) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Mout, MoutGPU, (n * p) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Mout2, MoutGPU2, (n * p) * sizeof(float), cudaMemcpyHostToDevice);
  
  
  cudaMatrixAdd<<<n, p>>>(d_M1, d_M2, d_Mout2, n, p);
  cudaMatrixMult<<<n,p>>>(d_M1, d_M2, d_Mout, n);
 
  
  
  cudaMemcpy(MoutGPU, d_Mout, n * p * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(MoutGPU2, d_Mout2, n * p * sizeof(float), cudaMemcpyDeviceToHost);
  
  
  printf("\nMatrice M1 + M2 (GPU) :\n");
  MatrixPrint(&MoutGPU2[0][0], n, p);
  
  printf("\nMatrice M1 * M2 (GPU) :\n");
  MatrixPrint(&MoutGPU[0][0], n, p);
  
    
  cudaFree(d_M1);
  cudaFree(d_M2);
  cudaFree(d_Mout);
  cudaFree(d_Mout2);
  
*/
  exit(EXIT_SUCCESS);
}
