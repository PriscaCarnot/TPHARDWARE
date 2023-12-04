#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <cuda_runtime.h>



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

void MatrixInit3D(float *M, int c, int n, int p, bool a){
 
 for (int i = 0; i< c; i++){
  for (int j = 0; j<n; j++){
   for (int k = 0; k<p ; k++){
    if (a == true){
     float number = rand();
     float randomValue = number / RAND_MAX;
     randomValue = 2*randomValue -1;
   
     //printf("Number %f \n",randomValue);
   
     *(M+i*n*p+j*p+k) = randomValue;
    }
    else{*(M+i*n*p+j*p+k) = 0;}
   }
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

__global__ void cudaMatrixConv(float *M1, float *M2, float *MoutGPU, int c, int n, int kernel){
  
  for (int i = 0;i<c;i++){
  // Parcours de raw_data
   for (int a = 0; a<n-kernel+1; a++ ){
    for (int b = 0; b<n-kernel+1; b++){
     // Parcours du noyau
     for (int k = 0; k<kernel; k++){
      for (int m = 0; m<kernel; m++){
       // Convolution
       float numberM1 = *(M1 +(a+k)*n+(b+m));
       float numberM2 = *(M2 +i*kernel*kernel+k*kernel+m);
       //printf("M1: %f, M2: %f \n",numberM1,numberM2);
       float numberOut = numberM1 * numberM2;
       *(MoutGPU +i*(n-kernel)*(n-kernel)+a*(n-kernel)+b) += numberOut;
      
     }
     
    }
    printf("Mres: %f \n",*(MoutGPU +i*(n-kernel)*(n-kernel)+a*(n-kernel)+b));
   }
  }
 }
}




int main() {
  int n = 4; // taille raw data
  int p = 3;
  int c = 6; 
  int l = 14;
  int m = 2;
  float  raw_data[n][n], C1_data[c][p][p], S1_data[c][l][l], C1_kernel[c][m][m];
  
  // Avec le CPU : 
  MatrixInit(&raw_data[0][0], n, n);
  //MatrixPrint(&raw_data[0][0], n, n);
  
  MatrixInit3D(&C1_data[0][0][0],c,p,p,false);
  MatrixInit3D(&S1_data[0][0][0],c,l,l,false);
  MatrixInit3D(&C1_kernel[0][0][0],c,m,m,true);
  
  MatrixPrint(&raw_data[0][0], n, n);
  MatrixPrint(&C1_kernel[0][0][0], m, m);
  
  
  // Avec le GPU : 
  float *d_raw_data, *d_C1_data, *d_S1_data, *d_C1_kernel;

  
  cudaMalloc((void**)&d_raw_data, (n * n) * sizeof(float));
  cudaMalloc((void**)&d_C1_data, (c*p*p) * sizeof(float));
  cudaMalloc((void**)&d_S1_data, (c*l*l) * sizeof(float));
  cudaMalloc((void**)&d_C1_kernel, (c*m*m) * sizeof(float));

    
  cudaMemcpy(d_raw_data, raw_data, (n * n) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C1_data, C1_data, (c*p*p) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_S1_data, S1_data, (c*l*l) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C1_kernel, C1_kernel, (c*m*m) * sizeof(float), cudaMemcpyHostToDevice);
  
  cudaMatrixConv<<<1, 1>>>(d_raw_data, d_C1_kernel, d_C1_data, c, n, m);
  //convolution_kernel<<<c,1>>>(d_raw_data, d_C1_kernel, d_C1_data);
  //cudaMatrixMult<<<n,p>>>(d_M1, d_M2, d_Mout, n);
 
  
  
  cudaMemcpy(C1_data, d_C1_data, c*p*p * sizeof(float), cudaMemcpyDeviceToHost);
  //cudaMemcpy(MoutGPU2, d_Mout2, n * p * sizeof(float), cudaMemcpyDeviceToHost);
  
  /*
  printf("\nMatrice M1 + M2 (GPU) :\n");
  MatrixPrint(&MoutGPU2[0][0], n, p);
  
  printf("\nMatrice M1 * M2 (GPU) :\n");
  MatrixPrint(&MoutGPU[0][0], n, p);
  */
    
  cudaFree(d_raw_data);
  cudaFree(d_C1_data);
  cudaFree(d_C1_kernel);
  cudaFree(d_S1_data);
  MatrixPrint(&C1_data[0][0][0], p, p);

  exit(EXIT_SUCCESS);
}
