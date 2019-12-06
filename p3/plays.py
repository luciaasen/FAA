import Datos as d
import numpy as np
import ClasificadorNew as cl
import EstrategiaParticionado as ep



def pruebaGenetica(dicc, porcentajes, tamsPob, gens, maxReglas, pElitismo, pCruce, repeticiones):
    cont = 0
    total = len(dicc) * len(porcentajes) * len(tamsPob) * len(gens) * len(maxReglas) * len(pElitismo) * len(pCruce) * repeticiones
    result_matrix = []
    for fileName in dicc:
        dataset = d.Datos(dicc[fileName])
        for prcnt in porcentajes:
            for tam in tamsPob:
                for epoca in gens:
                    for reg in maxReglas:
                        for pe in pElitismo:
                            for pc in pCruce:
                                errors = []
                                mejoresCr = []
                                numGens = []
                                avgFitness = []
                                gen = cl.ClasificadorGenetico(tamPoblacion=tam, nEpocas=epoca, pCruce=pc, pElit=pe, maxReglas=reg, usePrior=True)
                                for i in range(repeticiones):
                                    cont += 1
                                    print("Iteracion ",cont,"/", total)
                                    estrategia = ep.ValidacionSimple(prcnt)
                                    errors.append(gen.validacion(estrategia,dataset,gen))
                                    mejoresCr.append(gen.poblacion[-1])
                                    numGens.append(gen.currentGen)
                                    avgFitness.append(gen.avgFitness)
                                errorsnp = np.array(errors)
                                numGensnp = np.array(numGens)
                                avgFitnessnp = np.array(avgFitness)

                                mean,std = np.mean(errors), np.std(errors)
                                fitMean = np.mean(avgFitnessnp)
                                gensMean = np.mean(numGensnp)
                                result_matrix.append([dicc[fileName], prcnt, tam, epoca, reg, pe, pc, gensMean, fitMean, mean, std, mejoresCr])

    return result_matrix


dicc = {'ejemplo1' : './DatasetEjemplo/ejemplo1.data', 'ejemplo2' : './DatasetEjemplo/ejemplo2.data', 'tic' : './DatasetEjemplo/tic-tac-toe.data'}
porcentajes = [20, 40, 80]
tamsPob = [100, 200]
gens = [100, 300]
maxReglas = [1, 5, 10, 15, 20]
pElitismo = [0.05, 0.1, 0.15]
pCruce = [0.2, 0.4, 0.6, 0.8, 0.85]
repeticiones = 5

# dicc = dicc = {'ejemplo1' : './DatasetEjemplo/ejemplo1.data', 'ejemplo2' : './DatasetEjemplo/ejemplo2.data', 'tic' : './DatasetEjemplo/tic-tac-toe.data'}
# porcentajes = [80, 90]
# tamsPob = [10, 20]
# gens = [2, 4]
# maxReglas = [1, 3]
# pElitismo = [0.05, 0.1]
# pCruce = [0.2, 0.5]
# repeticiones = 3

# De la forma implementada, mutación es en función del elitismo y los cruces,
# de manera que hay pMutacion = 1 - pElitismo - pCruce
res = pruebaGenetica(dicc, porcentajes, tamsPob, gens, maxReglas, pElitismo, pCruce, repeticiones)
print(res)
