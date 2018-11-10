import numpy as np
from metodos_sistemas_lineares import LU

def minimos_quadrados(x,y,n):
    """
    
        Executa e retorna a função do método dos Mínimos Quadrados para o caso geral.
        
        Parametros:
        ------------
        x,y: Numpy arrays representando os valores dos pontos a serem utilizados.
        n: Inteiro representando os grau caracteristico dos pontos.
        
        Retorna:
        ------------
        function: Função lambda que representa a função de mínimos quadrados.
        
    """
    
    phi = []
    
    for i in range(n+1):
        aux = np.float128(np.array(x)**i)
        phi.append(aux)
    
    matriz = []
    for i in range(n+1):
        linha = []
        for j in range(n+1):
            linha.append(np.dot(phi[i],phi[j]))
        matriz.append(linha)
        
    #matriz = np.matrix(matriz, dtype=np.float64)
    
    b = [np.dot(y,p) for p in phi]
    
    b = np.array(b, dtype=np.float128)
    
    constantes = LU(matriz, b)[0] # [0] para selecionar somente o vetor de valores
    
    return lambda x: sum([np.float128((x**i)*c) for (c,i) in zip(constantes,range(n+1))])

def regressao_linear(x,y):
    """
    
        Executa e retorna a função a regressão linear dos pontos dados.
        
        Parametros:
        ------------
        x,y: Numpy arrays representando os valores dos pontos a serem utilizados.
        
        Retorna:
        ------------
        function: Função lambda que representa a função de regressão linear.
        
    """
    
    return minimos_quadrados(x,y,1)

    
def mmq_nao_linear(x,y):
    return minimos_quadrados(x,np.log(y),1)

def coeficiente_determinacao(x,y,p):
    numerador = sum(np.float128((y-p)**2))
    denominador = sum(np.float128(y**2)) - np.float128((1/len(y))*(sum(y))**2)
    
    r = np.sqrt(1 - numerador/denominador)
    
    return r