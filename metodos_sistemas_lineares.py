import numpy as np

def substituicao(matriz,vetorb):
    # Dimensão da matrix
    N = len(vetorb)
    # Vetor de incognitas
    x = np.array([np.float128(0)]*N)
    x[0] = vetorb[0]/matriz[0][0]

    for i in range(1,N):
        s = vetorb[i]
        
        for j in range(0,i):
            s = s - matriz[i][j] * x[j]
        
        x[i] = s / matriz[i][i]
    
    return np.float128(x)



def retro_substituicao(matriz,vetorb):
    # Dimensão da matrix
    N = len(vetorb)
    # Vetor de incognitas
    x = np.array([np.float128(0)]*N)
    x[N-1] = vetorb[N-1]/matriz[N-1][N-1]

    for i in np.arange(N-2,-1,-1):
        s = vetorb[i]
        
        for j in range(i,N):
            s -= matriz[i][j] * x[j]
        
        x[i] = s / matriz[i][i]
    
    return np.float128(x)



def Gauss(matriz,vetorb):
    # Dimensão da matrix
    N = len(vetorb)
    iteracoes = 0
    
    for k in range(0,N-1):
        iteracoes+=1
        for i in range(k+1,N):
            iteracoes+=1
            m = matriz[i][k] / matriz[k][k]
            
            for j in range(k, N):
                iteracoes+=1
                matriz[i][j] = matriz[i][j] - m * matriz[k][j]
                
            vetorb[i] = vetorb[i] - m * vetorb[k]
            
    x = retro_substituicao(matriz,vetorb)
    
    return x, iteracoes



def trocalinhas(matriz,B, linha1, linha2):
    aux = matriz[linha1].copy()
    matriz[linha1] = matriz[linha2]
    matriz[linha2] = aux
    
    aux = B[linha1]
    B[linha1] = B[linha2]
    B[linha2] = aux
    
    
def Gauss_pivot(matriz,vetorb):
    # Dimensão da matrix
    N = len(vetorb)
    iteracoes = 0

    for k in range(0,N-1):
        w = abs(matriz[k][k])
        
        lr = k
        for j in range(k,N):
            if abs(matriz[j][k]) > w:
                w = abs(matriz[j][k])
                lr = j
        
        if (lr != k):
            trocalinhas(matriz,vetorb,k,lr)
            
        
        for i in range(k+1,N):
            m = matriz[i][k] / matriz[k][k]
            iteracoes+=1
            for j in range(k,N):
                iteracoes+=1
                matriz[i][j] = matriz[i][j] - m * matriz[k][j]
                
                
            vetorb[i] = vetorb[i] - m * vetorb[k]
            
    x = retro_substituicao(matriz,vetorb)
    
    return x, iteracoes


def LU(A,B):
    N = len(A)
    U = np.array(A, dtype=np.float128)
    L = np.array([[np.float128(0)]*N]*N)
    iteracoes = 0
    
    for x in range(N):
        L[x][x] = np.float128(1)
    
    for i in range(N-1):
        iteracoes+=1
        for j in range(i+1,N):
            iteracoes+=1
            L[j][i] = U[j][i] / U[i][i]
            U[j] = U[j] - L[j][i] * U[i]
            U[j][i] = 0.0

    y = substituicao(L,B)
    x = retro_substituicao(U,y)
    
    return x, iteracoes



def Cholesky(A,B):
    n = len(B)
    G = np.array([[np.float64(0)]*n]*n)
    iteracoes = 0
    
    for j in range(n):
        iteracoes+=1
        s = sum([G[j][k]**2 for k in range(j)])
        iteracoes+=j-1
        G[j][j] = np.sqrt(A[j][j] - s)
        
        for i in range(j+1,n):
            iteracoes+=1
            s = sum([G[i][k]*G[j][k] for k in range(j)])
            iteracoes+=j-1
            G[i][j] = (A[i][j] - s)/G[j][j]
            
    y = substituicao(G,B)
    x = retro_substituicao(G.T,y)
    
    return x, iteracoes


def Jacobi(A,b,x0,maxit,erro):
    
    n = len(b)
    xk = x0
    xk1 = [np.float64(0)]*n
    iteracoes = 0
    
    for k in range(maxit):
        iteracoes+=1
        for i in range(n):
            s1 = sum([A[i][j]*xk[j] for j in range(i)])
            s2 = sum([A[i][j]*xk[j] for j in range(i+1,n)])
            iteracoes+=i-1+(i+n)
            xk1[i] = (b[i] - s1 - s2)/A[i][i]
        nmax = max([abs(xk1[i] - xk[i]) for i in range(n)])

        if nmax < erro:
            return np.array(xk1), iteracoes
        else:
            xk = xk1.copy()

            
            
def GaussSiedel(A,B, chute, maxit, erro):
    
    n = len(B)
    x = [chute]*n
    x_ant = [chute]*n
    iteracoes = 0
    
    for k in range(maxit):
        iteracoes+=1
        for i in range(n):
            s1 = sum([A[i][j]*x[j] for j in range(i)])
            s2 = sum([A[i][j]*x[j] for j in range(i+1,n)])
            iteracoes+=i-1+(i+n)
            x[i] = (B[i] - s1 - s2)/A[i][i]
        
        nmax = max([abs(x[i] - x_ant[i]) for i in range(n)])

        if nmax < erro:
            return np.array(x), iteracoes
        else:
            x_ant = x.copy()
            
            

def preenche_tridiagonal(a,b,c):
    n = len(b)
    
    M = np.array([[np.float64(0)]*n]*n) # Matriz tridiadonal a ser retornada
    
    for i in range(n):
        M[i][i] = b[i]
        if (i < n-1 and i < n-1):
            M[i][i+1] = a[i]
        if (i > 0 and i < n-1):
            M[i][i-1] = c[i]
            
    M[n-1][-2] = c[-1]
        
    return M



def Thomas(A,B):
    ''' Resolve Ax = d onde A é uma matriz tridiagonal composta pelos vetores a, b, c
    a - subdiagonal
    b - diagonal principal
    c - superdiagonal
    Retorna x
    '''
    
    a = [A[i][i+1] for i in range(len(A)-1)]
    b = [A[i][i] for i in range(len(A))]
    c = [A[i][i-1] for i in range(1,len(A))]
    d = B
    n = len(d)
    iteracoes = 0
    
    c_ = [ c[0] / b[0] ]
    d_ = [ d[0] / b[0] ]
    
    for i in range(1, n):
        iteracoes+=1
        aux = b[i] - c_[i-1]*a[i-1]
        if i < n-1:
            c_.append( c[i] / aux )
        d_.append( (d[i] - d_[i-1]*a[i-1])/aux )
    
    # Substituição de volta
    x = [d_[-1]]
    for i in range(n-2, -1, -1):
        iteracoes+=1
        x = [ d_[i] - c_[i]*x[0] ] + x
    
    return np.array(x), iteracoes



def critLinhas(A):
    alfa = []
    n = len(A)
    for i in range(n):
        s = 0
        for j in range(n):
            if i == j: continue
            s+=abs(A[i][j]/A[i][i])
        alfa.append(s)
    
    if (max(alfa) < 1):
        print('Converge: '+str(max(alfa))+' < 1')
    else:
        print('Não Converge: '+str(max(alfa))+' >= 1')