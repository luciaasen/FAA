def clasificaROC(datosTest, atributosDiscretos, diccionario, alpha):
    # A set with al classes
    classes = sorted(diccionario[-1].values())
    prob = dict()
    pr = np.zeros(len(datos))
    # For each data
    i = 0
    for dato in datosTest:
        pr[i] = dict()
        # And for each class
        for clase in classes:
            vero = 1
            j = 0
            # We calculate the product or all veros
            # of all attribute values in data, given the class
            for value in dato[:-1]:
                if atributosDiscretos[j]: #Nominal
                    nOccurrences = self.NBTables[j][clase][value]
                    vero *= nOccurrences/sum(self.NBTables[j][clase].values())
                else:#Discreto
                    vero *= self.NBTables[i][clase].pdf(value)
                j+=1
            pr[i][clase] = vero
        i+=1
    # Positive class = 1: we get the probability for all data
    # given the positive class, and normalize the vector
    positiveProbs = np.array([pr[i][1] for i in len(datosTest)])
    positiveProbs /= np.linalg.norm(positiveProbs)
    pred = [1 if prob < alpha else 0 for prob in positiveProbs]
    return pred
