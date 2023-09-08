import numpy as mp
import matplotlib as plt

def distance(x,y,xi,yi):
    return (x-xi)**2 + (y-yi)**2
data = [[1,1,1],[1,2,1],[2,1,1],[2,2,1],[2,2,1],[4,4,1],[4,5,1],[5,4,1],[5,5,1]]
k = 2
clusters = [[6,6],[0,0]]

iteraciones = 5

for i in range(iteraciones):
#Selecciona el cluster más cercano
    for j in range(len(data)):
        min = [1000000000,clusters[]]
        for k in range(clusters):
            datai = data[j]
            clusteri = clusters[k]
            min = min(min,distance(clusters[0],clusters[1],datai[0],datai[1]))
        

