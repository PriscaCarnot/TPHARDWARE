# TPHARDWARE
TP ENSEA 3A HARDWARE

## Compte rendu 

## Objectif du TP :

 - Implémentation du model LeNet5 avec cuda : 

![image](https://github.com/PriscaCarnot/TPHARDWARE/assets/120046244/8082df58-c952-4aa8-a936-bfeb7f892d80)

- Montrer que passer par le GPU est plus rapide que par le CPU pour des tailles de matrices importantes.

- Compréhension de cuda, des blocks et des threads

## Descriptifs des fichiers :  

CalculMatricielle.cu : Contient toutes les fonctions de bases de calculs matricielles (addition, multiplication).

CNN_Layer.cu : Contient les fonctions de convolutions entre matrice, ainsi que la fonction d'activation

printMNIST_1.cu : Contient la fonction main qui fait la pipeline de tout le réseaux de neuronnes. 

Pour lancer un fichier .cu : 
nvcc CalculMatricelle -o main
./main 


## Partie 1 : CalculMatricielle.cu 

Objectif de la partie : Comparer le temps de calculs de fonctions d'opérations de base entre matrices sur CPU puis GPU. 


Résultat calcul de base matricielles : 

![image](https://github.com/PriscaCarnot/TPHARDWARE/assets/118208053/f701d992-dc4f-4f12-9d4f-70efc47a8273)

Temps d'éxecution CPU et GPU : 

![image](https://github.com/PriscaCarnot/TPHARDWARE/assets/118208053/20b85af1-86aa-4515-a99d-a1feed5d32d8)

Action la plus chronophage : transfert de données entre le CPU et le GPU.

Conclusion : Pour des tailles de matrice élevées, le plus long est le temps de transfert des données entre le GPU et le CPU. Pour des tailles de matrices faible, le CPU est plus rapide. Lorsque l'on travaille avec des matrices de grandes tailles, passer par le GPU devient indispensable pour gagner du temps. 


## Partie 2 : CNN_layer.cu

Résultat :

Entrée : matrice de données de 8\*8, 

Noyau de convolution : matrice de 2\*5\*5, 

Matrice après convolution :  2\*4\*4 

Matrice après subsampling : 2\*2\*2

![image](https://github.com/PriscaCarnot/TPHARDWARE/assets/118208053/46d8f261-fab7-4b38-8843-a3449b6636e6)

![image](https://github.com/PriscaCarnot/TPHARDWARE/assets/118208053/d13650cb-f52f-479a-b06f-aee461d5f209)


Résultat : Nous avons vérifier la convolution pour des matrices de petites tailles. 
Là encore, le CPU est plus rapide pour les matrices de petites tailles (à cause du temps non négligeable de transfert de données entre le CPU et GPU). Dès que la taille augmente, le GPU devient plus efficace. 

## Partie 3 : printMNIST.cu

