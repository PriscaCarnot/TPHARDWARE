# TPHARDWARE
TP ENSEA 3A HARDWARE

## Compte rendu 

## Objectif du TP :
mémoire
 - Implémentation du model LeNet5 avec cuda : 

![image](https://github.com/PriscaCarnot/TPHARDWARE/assets/120046244/8082df58-c952-4aa8-a936-bfeb7f892d80)

- Montrer que passer par le GPU est plus rapide que par le CPU pour des tailles de matrices importantes.

- Compréhension de cuda, des blocks et des threads

## Descriptifs des fichiers :  

CalculMatricielle.cu : Contient toutes les fonctions de bases de calculs matricielles (addition, multiplication).

CNN_Layer.cu : Contient les fonctions de convolutions entre matrice, ainsi que la fonction d'activation

printMNIST_1.cu : Contient la fonction main qui fait la pipeline de tout le réseaux de neuronnes. 

Pour lancer un fichier .cu : 

nvcc CalculMatricelle.cu -o main

./main 


## Partie 1 : CalculMatricielle.cu 

Objectif de la partie : Comparer le temps de calculs de fonctions d'opérations de base entre matrices sur CPU puis GPU. 

### Implémentation GPU : 
- Définir variables et pointeursmémoire
- Alouer de la mémoire aux pointeurs avec cudaMalloc
- Envoyer variables et pointeurs du CPU au GPU avec cudaMemcpy
- Définir les dimensiosn des blocks et threads 
- Faire la fonction voulu : NomFonction<<<numBlock, Threadbyblocks*>>>(variable fonction)
- Envoyer les variables et pointeurs du GPU au CPU avec cudaMemcpy
- Libérer la mémoire alouée avec cudaFree

  *Doit être inférieur à 1024

Les threads permettent de paralléliser des tâches longues tel que lors de calcul matricielles pour les rendre plus rapide. Ici, pour ne pas être limité par la taille du ThreadSize, on utilise BlockSize pour diviser les calculs matricielles en plusieurs petits calculs fait par des threads (cela revient à utiliser plusieurs threads). 

On utilise blockIdx.x pour récupérer l'indice x (ou y ou z). Le but est de paralléliser au maximum les calculs indépendants pour augmenter la vitesse de calcul.

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


### Convolution : 

![image](https://github.com/PriscaCarnot/TPHARDWARE/assets/118208053/fef33c4c-a97a-4bf3-9b7b-6081d4c75e53)

Exécution (dans ce cas) de 2\*4\*4 threads pour obtenir la matrice de sortie. Un thread permet le calcul d'une convolution. 

Indices de convolutions : 

Matrice 1 : indice de la matrice du premier élément de convolution + indice du kernel (M1: 2 dimensions et M2: 3 dimensions)

Matrice 2 : indice du kernel

![image](https://github.com/PriscaCarnot/TPHARDWARE/assets/118208053/46d8f261-fab7-4b38-8843-a3449b6636e6)

![image](https://github.com/PriscaCarnot/TPHARDWARE/assets/118208053/d13650cb-f52f-479a-b06f-aee461d5f209)


Résultat : Nous avons vérifier la convolution pour des matrices de petites tailles. 
Là encore, le CPU est plus rapide pour les matrices de petites tailles (à cause du temps non négligeable de transfert de données entre le CPU et GPU). Dès que la taille augmente, le GPU devient plus efficace. 

### Sous échantillonnage :

Moyennage sur une matrice carré 2\*2 (soit une convolution avec un noyau constitué uniquement de coefficients 1/4 avec un stride de 2).

Pour ajuster le stride à 2, on multiplie l'indice x et y par 2 pour le moyennage (pas pour l'affectation des coefficients dans la matrice de sortie).

### Flatten : 

Coller les lignes de la matrice de bout en bout pour obtenir un vecteur.

### Dense layer : 

Multiplier le vecteur précédent par les poids de la couche et ajouter un biais. Les poids se trouve dans le notebook donné.

## Partie 3 : printMNIST.cu
Partie cuda : Après éxécution du code, deux images de 5 s'affiche (l'image et son label associé). Cela correspond aux données d'entrées. L'image d'entrée doit passer par les différentes couches du réseaux. Le but étant d'obtenir à la fin son label. 
Partie python : Nous avons récupérer les deux fichiers après exécution du code mais nous n'avons pas réussi à utiliser les éléments contenu dans le fichier .h5.
