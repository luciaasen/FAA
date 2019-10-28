from Datos import Datos
from sklearn.model_selection import train_test_split, KFold
from sklearn import preprocessing
KBINSDISCRETISER
# from sklearn.preprocessing import OneHotEncoder

# Variables al principio para facilitar ejecucion
porcentajeValidacionSimple = 0.2
seed = 0

# dataset=Datos('datos/tic-tac-toe.data')
# dataset=Datos('datos/conjunto_datos_lentillas.txt')
dataset=Datos('datos/german.data')

# Para que funcione con scikit hay que discretizar los atributos continuos
# utilizaremos OneHotEncoder del modulo preprocessing de sklearn
encAtributos = preprocessing.OneHotEncoder(categorical_features=dataset.nominalAtributos[:-1], sparse=False)
encMatrix = encAtributos.fit_transform(dataset.datos[:,:-1])
clases = dataset.datos[:,-1]

print("Original Matrix\n", dataset.datos)

print("encAtributos:\n", encAtributos)
print("encMatrix:\n", encMatrix)
print("clases:\n", clases)
###############################################################################
###############################################################################
# ValidacionSimpleScikit
###############################################################################
###############################################################################
# Particion de los datos

train, test = train_test_split(encMatrix, test_size=porcentajeValidacionSimple, random_state=seed, shuffle=True)
print("Train: \n", train)
print("Test: \n", test)
