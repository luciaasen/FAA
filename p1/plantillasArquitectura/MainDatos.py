# -*- coding: utf-8 -*-
"""

@author: profesores faa
"""

from Datos import Datos
import EstrategiaParticionado

# dataset=Datos('datos/tic-tac-toe.data')
dataset=Datos('datos/conjunto_datos_lentillas.txt')
datos_mat = dataset.datos
print(datos_mat)

particionadoSimple = EstrategiaParticionado.ValidacionSimple(50)
part1 = particionadoSimple.creaParticiones(datos_mat)

print("Longitud lista de particiones 1 = " + str(len(part1)))
cont = 0
for i in part1:
    print("iteracion " + str(cont) + ":")
    print("     Train: " + str(i.indicesTrain))
    print("     Test: " + str(i.indicesTest))
    cont = cont + 1
# print("conjunto Train de part1 tiene " + str(len(part1.indicesTrain)) + " elementos")
# print("conjunto Test de part1 tiene " + str(len(part1.indicesTest)) + " elementos")


particionadoCruzado = EstrategiaParticionado.ValidacionCruzada(10)
part2 = particionadoCruzado.creaParticiones(datos_mat)

print("Longitud lista de particiones 2 = " + str(len(part2)))
cont = 0
for i in part2:
    print("iteracion " + str(cont) + ":")
    print("     Train: " + str(i.indicesTrain))
    print("     Test: " + str(i.indicesTest))
    cont = cont + 1
# print("conjunto Train de part2 tiene " + str(len(part2.indicesTrain)) + " elementos")
# print("conjunto Test de part2 tiene " + str(len(part2.indicesTest)) + " elementos")
# dataset=Datos('datos/german.data')
