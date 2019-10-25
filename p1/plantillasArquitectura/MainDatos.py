# -*- coding: utf-8 -*-
"""

@author: profesores faa
"""

from Datos import Datos
import EstrategiaParticionado

dataset=Datos('datos/tic-tac-toe.data')
datos_mat = dataset.datos

particionadoSimple = EstrategiaParticionado.ValidacionSimple("ValidacionSimple", 50)
part1 = particionadoSimple.creaParticiones(datos_mat)

print("conjunto Train de part1 tiene " + str(len(part1.indicesTrain)) + " elementos")
print("conjunto Test de part1 tiene " + str(len(part1.indicesTest)) + " elementos")


particionadoCruzado = EstrategiaParticionado.ValidacionCruzada("ValidacionCruzada", 10)
part2 = particionadoCruzado.creaParticiones(datos_mat)

print("conjunto Train de part2 tiene " + str(len(part2.indicesTrain)) + " elementos")
print("conjunto Test de part2 tiene " + str(len(part2.indicesTest)) + " elementos")
# dataset=Datos('datos/german.data')
