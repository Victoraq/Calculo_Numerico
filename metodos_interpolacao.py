import numpy as np

def lagrange(n, x, y, z):
    final = np.array([0.0]*len(z)) # vetor de saida

    for k in range(len(z)):
        r = 0.0
        for i in range(n):
            c = 1
            d = 1
            for j in range(n):
                if i != j:
                    c = c * (z[k]-x[j])
                    d = d * (x[i] - x[j])

            r = r + y[i]*(c/d)

        final[k] = r

    return final

def norma_max(analitica, estimado):
    max = abs(analitica[0] - estimado[0])
    for (real, est) in zip(analitica, estimado):
        erro = abs(real - est)
        if erro > max: max = erro
    return max

def linear_partes(n, x, y, z):
    final = [] # vetor de saida
    
    # Percorrendo todos os valores conhecidos
    for i in range(1,n):
        # valores que serão interpolados entre x[i-1] e x[i]
        valores = z[(x[i-1] <= z)&(z < x[i])]
        
        if (i == n-1): valores = z[(x[i-1] <= z)&(z <= x[i])]
        
        # interpolando por retas os pontos passados
        for k in valores:
            # Formando a reta y = ax + b
            a = np.float64((y[i]-y[i-1])/(x[i]-x[i-1]))
            b = y[i] - a*x[i]

            # Descobrindo o ponto a ser interpolado
            v = np.float64(a*k + b)
            
            final.append(v)
    

    return np.array(final)