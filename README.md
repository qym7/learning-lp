![logo RTE](https://webmail.polytechnique.fr/service/home/~/?auth=co&loc=fr_FR&id=7045&part=2)
# Apport des réseaux de neurones à la résolution approchée de problèmes d’optimisation linéaire

---
### Objectifs
	

	
L’objectif primaire du projet sera d’évaluer dans quelle mesure il est réalisable d’entraîner un réseau de neurones capable de déterminer des solutions suffisamment précises de certains problèmes d’optimisation linéaire. 

Alors qu’il s’agira donc tout d’abord d’une étude de faisabilité, le groupe se proposera en second lieu de fournir un tel réseau, à condition que le premier objectif soit atteint. 

Enfin, le groupe avancera dans son étude en élaborant progressivement une interface conviviale permettant l’entraînement automatique d’un réseau de neurones adapté à un problème d’optimisation linéaire fourni en entrée. 


### Situation de départ 

Le projet s’inscrit dans la continuité du travail « Apport du machine learning pour la résolution approchée de problèmes d’optimisation linéaire », effectué par Eulalie Creusé, Étienne Maeght, Yiming Qin, Arthur Schichl et Kunhao Zheng. 

Celui-ci s’achève sur plusieurs résultats très encourageants concernant des problèmes d’optimisation linéaire de petite dimension. Cependant, l’étude met en évidence deux difficultés majeures qui apparaissent lorsque la taille du réseau et le nombre de problèmes croit, soient la perte de régularité des fonctions coûts des réseaux (en particulier l’apparition de minima locaux non optimaux) et l’augmentation rapide du temps de calcul.


### Démarche détaillée

Le groupe cherchera à atteindre ses objectifs en surmontant les difficultés évoquées précédemment. 

Sa première priorité sera de réfléchir à des moyens d’échapper à des minima locaux non optimaux de la fonction coût, notamment en réfléchissant à de nouvelles manières de faire varier la learning rate au cours des époques de l’entraînement. 

Après quelques semaines, on procédera à une évaluation approfondie des résultats obtenus afin de définir les voies pour la suite et discuter la pertinence d’une priorisation de la diminution du temps de calcul des codes par plusieurs approches.
	
La démarche détaillé est présentée dans le **document « démarche détaillée »**.