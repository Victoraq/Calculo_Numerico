import pesos
import abscissa
import numpy as np

w = pesos.pesos() # Pesos da quadratura de gauss
t = abscissa.abscissas() # Abscissas da quadratura de gauss

def quadratura_gauss(f,a,b,n_pontos):
    # leio o pesos e as abssisas a partir de n
    ordem = 2*n_pontos - 1
    wn = w[n_pontos]
    tn = t[n_pontos]
    integral = np.float128(0.0)
    
    # mudança de variável
    xt = lambda t: t*(b-a)/2.0 + (b+a)/2.0
    x_linha = (b-a)/2.0
    
    for i in range(n_pontos):
        integral += np.float128(wn[i] * f(xt(tn[i])) * x_linha)
        
    return integral

def retangulo(f, a, b):
    h = b - a
    
    return h * f(a)
    
def ponto_medio(f, a, b):
    h = b - a
    
    fy = f((a + b)/2)
    
    return h * fy

def trapezio(f, a, b):
    h = b - a
    
    return h/2*(f(a) + f(b))

def simpson1_3(f, a, b):
    h = (b - a)/2
    x1 = (a + b)/2
    
    valor = h/3 * (f(a) + 4*f(x1) +f(b))
    
    return valor

def simpson3_8(f, a, b):
    h = (b - a)/3
    
    x1 = a+h
    x2 = x1+h
    
    valor = (3*h)/8 * (f(a) + 3*f(x1) + 3*f(x2) + f(b))
    
    return valor    

def repetida(f, a, b, n, metodo):
    h = (b - a)/n
    valor = 0
    for i in range(1,n+1):
        valor += metodo(f,a + (h * (i-1)), a + h*i)
    
    return valor