#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>



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

void MatrixPrint3D(float *M, int n, int p, int c){
  for (int i = 0; i< c; i++){
   for (int j = 0; j<n; j++){
    for (int k = 0; k<p ; k++){
      printf(" %f \t",*(M+i*n*p+j*p+k));
     
   }
   printf("\n");
  }
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

__device__ float activation_tanh(float M){
 return tanhf(M);
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


// Convolution sur GPU
__global__ void cudaMatrixConv(float *M1, float *M2, float *MoutGPU, int c, int n, int kernel){
  
     int i = blockIdx.x;
     int a = blockIdx.y;
     int b = blockIdx.z;
     
     // Parcours du noyau
     float num = 0;
     for (int k = 0; k<kernel; k++){
      for (int m = 0; m<kernel; m++){
       // Convolution
       float numberM1 = *(M1 +(a+k)*n+(b+m));
       float numberM2 = *(M2 +i*kernel*kernel+k*kernel+m);
       float numberOut = numberM1 * numberM2;
       num += numberOut;
      
     }
    }
    *(MoutGPU +i*(n-kernel+1)*(n-kernel+1)+a*(n-kernel+1)+b) = num;
}


// Moyenneur = Convolution avec un filtre [[1 ,1],[1, 1]]
__global__ void cudaMatrixSubSamp(float *M1, float *MoutGPU, int c, int p, int l){

     int i = blockIdx.x;
     int a = blockIdx.y*2;
     int b = blockIdx.z*2;
     // Parcours du noyau
     float num = 0;
     for (int k = 0; k<2; k++){
      for (int m = 0; m<2; m++){
       // Convolution
       float numberOut = *(M1 +i*p*p+(a+k)*p+(b+m));
       num += numberOut;
     }
    }
    *(MoutGPU +i*(l)*(l)+a/2*(l)+b/2) = activation_tanh(num/4);
}


__global__ void cudaFlatten(int n, int p, float* input, float* output) {
    int i = blockIdx.x;
    int j = blockIdx.y;
    int z = blockIdx.z;
    for (int k = 0; k<n; k++){
     for (int l = 0; l<p; l++) {
      *(output+i*n*p+j*p+z) = *(input+i*n*p+j*p+z);
      }
    }
}

int cudaDense(){
 FILE *fptr;

 if((fptr = fopen("FashionMNIST_weights.h5","rb")) == NULL){
    printf("Can't open file");
    exit(1);
  }
  /*
  fread(&magic, sizeof(int), 1, fptr);
  fread(&nbImg, sizeof(int), 1, fptr);
  fread(&nbRows, sizeof(int), 1, fptr);
  fread(&nbCols, sizeof(int), 1, fptr);
  
  printf("Nb Magic : %u \n", magic);
  printf("Nb Img : %u \n", nbImg);
  printf("Nb Rows : %u \n", nbRows);
  printf("Nb Cols : %u \n", nbCols);
  */
  
  exit(0);
}

int main1() {
  int n = 8; // taille raw data
  int p = 4; // taille matrice sortie 1ere convolution
  int c = 2; // Nombre de kernel
  int l = 2; // taille après 1er sous échantillonnage
  int m = 5; // taille du kernel
  float  raw_data[n][n], C1_data[c][p][p], S1_data[c][l][l], C1_kernel[c][m][m];
  
  // Avec le CPU : 
  MatrixInit(&raw_data[0][0], n, n);
  //MatrixPrint(&raw_data[0][0], n, n);
  
  MatrixInit3D(&C1_data[0][0][0],c,p,p,false);
  MatrixInit3D(&S1_data[0][0][0],c,l,l,false);
  MatrixInit3D(&C1_kernel[0][0][0],c,m,m,true);
  
  printf("Matrice d'entree : \n");
  MatrixPrint(&raw_data[0][0], n, n);
  printf("Noyau de convolution : \n");
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
  
  dim3 gridSize(c, p, p);
  dim3 gridSize2(c, l, l);
  cudaMatrixConv<<<gridSize, 1>>>(d_raw_data, d_C1_kernel, d_C1_data, c, n, m); 
  cudaMatrixSubSamp<<<gridSize2, 1>>>(d_C1_data, d_S1_data, c, p, l);
  //cudaDense<<<1,1>>>();
  
  cudaMemcpy(C1_data, d_C1_data, c*p*p * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(S1_data, d_S1_data, c*l*l * sizeof(float), cudaMemcpyDeviceToHost);
    
  printf("Matrice après conv : \n");
  MatrixPrint3D(&C1_data[0][0][0], p, p, c);
  printf("Matrice après SE : \n");
  MatrixPrint3D(&S1_data[0][0][0], l, l, c);
  
  
  
  cudaFree(d_raw_data);
  cudaFree(d_C1_data);
  cudaFree(d_C1_kernel);
  cudaFree(d_S1_data);

  exit(EXIT_SUCCESS);
}
