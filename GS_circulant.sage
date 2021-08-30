import random
import cmath
import math
import numpy as np
import time

pi2j = cmath.pi*2j

def FT(f, q):
    #input: f, output, hat f over Z_q
    FT_array = np.array( [ 0.0 for i in range(q)] )
    for z in range(q):
        Wz = 0
        for x in range(q):
            Wz += f[x] * cmath.exp(pi2j*float(x*z)/q).real / math.sqrt(q) #* sinc_square_pdf(float(r), 4.0)
        FT_array[z] = Wz
    return FT_array

def DFT_of_uniform_state_matrix(B_left, B_right, q, power):
    """ Let the error state be the DFT of uniform, namely sinc. 
        If power >1, then it is (sinc)^power, i.e., the DFT of convolutions of uniform  """
    FTuniform = DFT_uniform(B_left, B_right, q)
    for i in range(q):
        FTuniform[i] = FTuniform[i]**power

    FTuniform = FTuniform/FTuniform.norm()
    print("DFT of bounded uniform distribution", FTuniform)

    M = Matrix(RR, [[ FTuniform[(i-j)%q] for i in range(q)] for j in range(q)])
    print(B_left, B_right, q)

    return M

def super_Gaussian_pdf(x, p, sigma):
    """  exp(- |x|^p )  """
    x = float(x) / sigma
    return math.exp(-(x*x)**(p/2.0))

def super_Gaussian_shift_n(x, p, sigma, q, pre):
    # simulating super-Gaussian over integer mod q
    z = 0.0
    for i in range (-pre, pre):
        z+=super_Gaussian_pdf(x+i*q, p, sigma)
    return z

def super_Gaussian_error_state_matrix(sigma, p, q):
    each_array = vector(RR, [ super_Gaussian_shift_n(i, p, sigma, q, 6) for i in range(q)])
    each_array = each_array/each_array.norm()
    print("state:", each_array)
    M = Matrix(RR,[[ each_array[(i-j)%q] for i in range(q)] for j in range(q)])
    print("super-Gaussian with sigma = ", sigma, "p = ", p, "q = ", q)
    return M

# Producing bounded uniform LWE error state. may not be useful for getting meaningful SIS bounds
def uniform_error_state_matrix(B, n):
    M = Matrix(RR,[[((i-j)%n<B)/math.sqrt(B) for i in range(n)] for j in range(n)])
    print("bounded uniform state with B, q = ", B, n)
    return M

#The classical Gran-Schmidt algorithm (in truth a very common variant, known for its stability)
#This algorithm performs operations over the field F. One can typically take F to be QQ or RR.
def Stable_GS(A,F):
    N = A.nrows();
    M = A.ncols();
    B = matrix(F,N,M);
    v = vector(F,N);
    for i in range(N):
        B[i] = A[i];
        for j in range(i):
            B[i] = B[i] - (B[i]*B[j])/(v[j])*B[j];
        v[i] = B[i]*B[i];
    return B;

def GS_length(q, G):
    print("x,   (G[x].norm())^2  " )
    for x in range(q):
        print(x, RR(G[x].norm())^2 )


for q in range(17, 18):
  for B in range(5, 6):
    start = time.time()

    M = super_Gaussian_error_state_matrix(B, 2.0, q)
    #M = DFT_of_uniform_state_matrix(-B, B, q, 1)
    #M = uniform_error_state_matrix(B, q)
    
    G = Stable_GS(M, RR)
    GS_length(q, G)

    end = time.time()
    print("Time of computing GS:", end - start)






