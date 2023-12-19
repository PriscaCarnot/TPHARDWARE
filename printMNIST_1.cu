#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "CNN_Layer.cu"

#define WIDTH 28
#define HEIGHT 28

void charBckgrndPrint(char *str, int rgb[3]){
  printf("\033[48;2;%d;%d;%dm", rgb[0], rgb[1], rgb[2]);
  printf("%s\033[0m",str);
}

void MatrixCopy(float *M, int *Mcopy, int n , int p){
 for (int i = 0; i< n; i++){
  for (int j = 0; j<p; j++){
    *(M+i*p+j) = (float) *(Mcopy+i*p+j);
  }
 }
}

void imgColorPrint(int height, int width, int ***img){
  int row, col;
  char *str="  ";
  for(row=0; row<height; row++){
    for(col=0; col<width; col++){
      charBckgrndPrint(str,img[row][col]);
    }
    printf("\n");
  }
}

int main() {
  int i, j;
  int ***img;
  int color[3]={255,0,0};
  unsigned int magic, nbImg, nbRows, nbCols;
  unsigned char val;
  FILE *fptr;

  // Malloc image
  img = (int ***)malloc(HEIGHT*sizeof(int **));
  for(i=0; i<HEIGHT; i++){
    img[i]= (int **)malloc(WIDTH*sizeof(int *));
    for(j=0; j<WIDTH; j++){
      img[i][j] = (int *)malloc(sizeof(int)*3);
    }
  }

  //Open File
  if((fptr = fopen("train-images.idx3-ubyte","rb")) == NULL){
    printf("Can't open file");
    exit(1);
  }

  //Read File
  fread(&magic, sizeof(int), 1, fptr);
  fread(&nbImg, sizeof(int), 1, fptr);
  fread(&nbRows, sizeof(int), 1, fptr);
  fread(&nbCols, sizeof(int), 1, fptr);
/*
  printf("Nb Magic : %u \n", magic);
  printf("Nb Img : %u \n", nbImg);
  printf("Nb Rows : %u \n", nbRows);
  printf("Nb Cols : %u \n", nbCols);
*/
  for(i=0; i<HEIGHT; i++){
    for(j=0; j<WIDTH; j++){ 
      fread(&val, sizeof(unsigned char), 1, fptr);  
      img[i][j][0]=(int)val*color[0]/255;
      img[i][j][1]=(int)val*color[1]/255;
      img[i][j][2]=(int)val*color[2]/255;
    }
  }

  imgColorPrint(HEIGHT, WIDTH, img);
/*
  // setup image grayscale
  for(i=0; i<HEIGHT; i++){
    for(j=0; j<WIDTH; j++){
        img[i][j][0] = ((i+j)*4)%255;
        img[i][j][1] = ((i+j)*4)%255;
        img[i][j][2] = ((i+j)*4)%255;
    }
  }
*/
  // print image
  imgColorPrint(HEIGHT, WIDTH, img);
  
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Partie CNN_LAYER : 
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  int n = 28; // taille raw data
  int m = 5; // taille du kernel
  int p = n-m+1; // taille matrice sortie 1ere convolution
  int c = 6; // Nombre de kernel
  int l = p/2; // taille après 1er sous échantillonnage  
  float C1_data[c][p][p], S1_data[c][l][l], C1_kernel[c][m][m];
  float raw_data[HEIGHT][WIDTH];
  float output[c*l*l];
  MatrixCopy(&raw_data[0][0], &img[0][0][0], HEIGHT, WIDTH);

  // Avec le CPU : 
  //MatrixInit(&raw_data[0][0], n, n);
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
  float *d_output;
  
  cudaMalloc((void**)&d_raw_data, (n * n) * sizeof(float));
  cudaMalloc((void**)&d_C1_data, (c*p*p) * sizeof(float));
  cudaMalloc((void**)&d_S1_data, (c*l*l) * sizeof(float));
  cudaMalloc((void**)&d_C1_kernel, (c*m*m) * sizeof(float));
  cudaMalloc((void**)&d_output, (c*l*l) * sizeof(float));
    
  cudaMemcpy(d_raw_data, raw_data, (n * n) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C1_data, C1_data, (c*p*p) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_S1_data, S1_data, (c*l*l) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C1_kernel, C1_kernel, (c*m*m) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_output, output, (c*l*l) * sizeof(float), cudaMemcpyHostToDevice);
  
  dim3 gridSize(c, p, p);
  dim3 gridSize2(c, l, l);
  dim3 gridSize3(c, l, l);
  
  cudaEvent_t startGPU,endGPU;
  cudaEventCreate(&startGPU);
  cudaEventCreate(&endGPU);
  
  
  cudaEvent_t startGPU2,endGPU2;
  cudaEventCreate(&startGPU2);
  cudaEventCreate(&endGPU2);
  
  
  cudaEvent_t startGPU3,endGPU3;
  cudaEventCreate(&startGPU3);
  cudaEventCreate(&endGPU3);
  
  cudaEventRecord(startGPU);
  cudaMatrixConv<<<gridSize, 1>>>(d_raw_data, d_C1_kernel, d_C1_data, c, n, m); 
  cudaEventRecord(endGPU);
  cudaEventSynchronize(endGPU);
  float GPUtime;
  cudaEventElapsedTime(&GPUtime, startGPU, endGPU);
  printf("\n Temps d'éxécution GPU (convolution): %f msec \n",GPUtime);
  
  cudaEventRecord(startGPU2);
  
  
  cudaMatrixSubSamp<<<gridSize2, 1>>>(d_C1_data, d_S1_data, c, p, l);
  cudaEventRecord(endGPU2);
  cudaEventSynchronize(endGPU2);
  float GPUtime2;
  cudaEventElapsedTime(&GPUtime2, startGPU2, endGPU2);  
  printf("\n Temps d'éxécution GPU (Sous-échantillonnage): %f msec \n",GPUtime2);
  
  cudaFlatten<<<gridSize3,1>>>(l, l,d_S1_data,  d_output);
  cudaDense();
  
  cudaMemcpy(C1_data, d_C1_data, c*p*p * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(S1_data, d_S1_data, c*l*l * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(output, d_output, c*l*l * sizeof(float), cudaMemcpyDeviceToHost);
    
  
  printf("Matrice après conv : \n");
  MatrixPrint3D(&C1_data[0][0][0], p, p, c);
  
  printf("Matrice après SE : \n");
  MatrixPrint3D(&S1_data[0][0][0], l, l, c);
  
  printf("haha\n");
  for (int i = 0; i<400; i++){
  	printf("%f\t", output[i]);
  	}
  
  
  cudaFree(d_raw_data);
  cudaFree(d_C1_data);
  cudaFree(d_C1_kernel);
  cudaFree(d_S1_data);
  exit(EXIT_SUCCESS);

}
