import numpy as np
from metodos_sistemas_lineares import Cholesky
from metodos_integracao_numerica import quadratura_gauss

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

def __func_prod(f,g,grauf,graug):
    return lambda x: f(x,grauf) * g(x,graug)

def phi(x,grau):
    return x**grau;

def prod_phi(phi, grau1, grau2):
    return lambda x: phi(x,grau1) * phi(x,grau2)

def minimos_quadrados_cont(f,a,b,n):
    """
    
        Executa e retorna a função do método dos Mínimos Quadrados para o caso contínuo.
        
        Parametros:
        ------------
        f: Função a ser interpolada.
        a,b: Inicio e fim do intervalo de integração.
        n: Inteiro representando a ordem de integração.
        
        Retorna:
        ------------
        function: Função lambda que representa a função de mínimos quadrados.
        
    """
    
    colunaB = []
    
    matriz = []
    for i in range(n+1):
        linha = []
        for j in range(n+1):
            prod = prod_phi(phi, i, j)
            integral = quadratura_gauss(prod,a,b,n+1) # realizando a integral do produto de funções
            
            linha.append(integral) 
        matriz.append(linha)
    
    for i in range(n+1):
        prod = lambda x: f(x)*phi(x,i) # realizando o produto entre as funções
            
        integral = quadratura_gauss(prod,a,b,n+1) # realizando a integral do produto de funções
        
        colunaB.append(integral)
    
    colunaB = np.array(colunaB, dtype=np.float128)
    
    constantes = Cholesky(matriz, colunaB)[0] # [0] para selecionar somente o vetor de valores

    return lambda x: sum([np.float128((x**i)*c) for (c,i) in zip(constantes,range(n+1))])