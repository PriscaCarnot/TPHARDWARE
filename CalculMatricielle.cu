#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>




// Initialise la matrice entre -1 et 1 
void MatrixInit (float *M, int n, int p){
 
 for (int i = 0; i< n; i++){
  for (int j = 0; j<p; j++){

   float number = rand();
   float randomValue = number / RAND_MAX;
   randomValue = 2*randomValue -1;
   
   //printf("Number %f \n",randomValue);
   
   *(M+i*p+j) = randomValue;
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



int main(int argc, char *argv[]) {
  int n = atoi(argv[1]);
  int p = atoi(argv[2]); 
  float  M1[n][p], M2[n][p], Mout[n][p], MoutGPU[n][p], MoutGPU2[n][p];
  
  // Avec le CPU : 
  MatrixInit(&M1[0][0], n, p);
  MatrixInit(&M2[0][0], n, p);
  MatrixInit(&Mout[0][0], n, p);
  MatrixInit(&MoutGPU[0][0], n, p);
  
  /*
  
  printf("Matrice M1 : \n");
  MatrixPrint(&M1[0][0], n, p);
  
  printf("Matrice M2 : \n");
  MatrixPrint(&M2[0][0], n, p);
  
  printf("\n");
  
  MatrixAdd(&M1[0][0], &M2[0][0], &Mout[0][0], n, p);
  printf("Matrice M1 + M2 : \n");
  MatrixPrint(&Mout[0][0], n, p);
  
  MatrixMult(&M1[0][0], &M2[0][0], &Mout[0][0], n);
  printf("\nMatrice M1 * M2 : \n");
  MatrixPrint(&Mout[0][0], n, p);
  */
  
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
  cudaMatrixMult<<<n, p>>>(d_M1, d_M2, d_Mout, n);
    
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

  exit(EXIT_SUCCESS);
}
