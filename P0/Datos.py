import numpy as np

class Datos:
  
  TiposDeAtributos=('Continuo','Nominal')
 
  def __init__(self, nombreFichero):
    f = open( nombreFichero, "r")
    nDatos = int(f.readline()) # Unused?
    #Read attributes names line and store it
    self.nombreAtributos = f.readline()[:-1].split(',')
    
    #Read attributes types line and store it. Check for invalid types.
    self.tipoAtributos = f.readline()[:-1].split(',')
    
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
    #TODO Crear diccionarios
    self.diccionarios = []
    
    
    print(self.datos[:3])
    f.close()
    
  # TODO: implementar en la práctica 1
  def extraeDatos(self, idx):
    pass
