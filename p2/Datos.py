import numpy as np

class Datos:

  TiposDeAtributos=('Continuo','Nominal')

  def __init__(self, nombreFichero, oneHot = False):
    f = open( nombreFichero, "r")
    nDatos = int(f.readline()) # Unused?
    #Read attributes names line and store it
    self.nombreAtributos = f.readline()[:-1].split(',')

    #Read attributes types line and store it. Check for invalid types.
    self.tipoAtributos = f.readline()[:-1].split(',')

    # Check that the attributes in the dataset match the attributes we expect
    for tipo in self.tipoAtributos:
        if tipo not in self.TiposDeAtributos:
            raise ValueError(tipo)

    # List containing True in the ith position if the ith atribute is nominal, False otherwise
    self.nominalAtributos = [True if i == 'Nominal' else False for i in self.tipoAtributos ]

    # We create a list whose elements are lists containing the data, and we get a no.array from it
    datos = []
    for line in f:
        datos.append(line[:-1].split(','))
    self.datos = np.array(datos)

    # For each column (attribute) of the matrix we need to create a dictionary
    self.diccionarios = []
    i = 0;
    for att in self.nominalAtributos:
        temp_dic = {}
        if att == True:
            #if the attribute is Nominal create a list with the ith column
            #of the array "datos"
            column = self.datos[:, i]
            #Now we create a set out of each column
            #This automatically orders the elements alphabetically
            column_set = sorted(set(column))

            #Now we see which elements belong to the set
            #and we add it to the dictionary
            temp_dic = {}
            value = 0
            for item in column_set:
                temp_dic[item] = value
                value = value + 1
        #if the attribute is Continuous append an empty dictionary
        self.diccionarios.append(temp_dic)
        i = i + 1

    #Now we have the list of dictionaries we have to modify the
    #matrix Datos and change it's values to the corresponding numbers in
    #our dictionaries for each column
    #For each column
    nAtributos = len(self.nominalAtributos)
    for col in range(0, nAtributos):
        #if the attribute is nominal
        if self.nominalAtributos[col] == True:
            #We replace each value with the corresponding value of the
            #corresponding dictionary
            for row in range(0, nDatos):
                self.datos[row, col] = self.diccionarios[col][self.datos[row, col]]

    #We change the matrix datatype to Float
    self.datos = self.datos.astype(dtype=float)
    
    if oneHot == True:
        # Create new matrix with desired dimensions
        newNCols = sum([max(1, len(diccionario)) for diccionario in self.diccionarios])
        newDatos = np.empty([len(datos), newNCols])

        # initialize a new atributos array
        # Maybe in the future it is also necessary to update other self.thingies?
        # self.dictionaries would be an array of empty dicts, idk. It might be necessary at some point
        nombreAtributos = []
        # Lets update nominalAtributos as well.
        nominalAtributos = []

        newi = 0
        for oldi in range(len(self.nominalAtributos)):
            if self.nominalAtributos[oldi] == False:
                # update matrix
                newDatos[:, newi] = self.datos[:, oldi]
                # update nombreAtributos, nominalAtributos
                nombreAtributos.append(self.nombreAtributos[oldi])
                nominalAtributos.append(self.nominalAtributos[oldi])
                newi += 1
            else:
                for key in self.diccionarios[oldi].keys():
                    # update matrix
                    value = self.diccionarios[oldi][key]
                    newcol = [1 if self.datos[k, oldi] == value else 0 for k in range(len(self.datos))]
                    newDatos[:, newi] = newcol 
                    # update nombreAtributos, nominalAtributos
                    nombreAtributos.append(self.nombreAtributos[oldi] + '_' + key) 
                    nominalAtributos.append(True)
                    newi += 1
        self.datos = newDatos
        self.nombreAtributos = nombreAtributos
        self.nominalAtributos = nominalAtributos
    # print(self.datos)
    f.close()

  #Recibe en idx una lista de indices que extraer y devolver de la
  #Matriz datos
  def extraeDatos(self, idx):
      return self.datos[idx]
