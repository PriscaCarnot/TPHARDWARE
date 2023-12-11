# TPHARDWARE
TP ENSEA 3A HARDWARE


TP du 27 novembre 2023
Prise en main de Cuda


Compte rendu 

nvcc CalculMatricelle -o main
./main 


Calcul de base de matrice : 

![image](https://github.com/PriscaCarnot/TPHARDWARE/assets/118208053/f701d992-dc4f-4f12-9d4f-70efc47a8273)

Temps d'éxecution CPU et GPU : 

![image](https://github.com/PriscaCarnot/TPHARDWARE/assets/118208053/20b85af1-86aa-4515-a99d-a1feed5d32d8)

Action la plus chronophage : transfert de données entre le CPU et le GPU.

Résultat convolution avec une matrice de données de 8**8, un noyau de convoluton de 2*5*5, une matrice de sortie à la premère convolution de 2*4*4 et une matrice de sortie après subsampling de 2*2*2 
