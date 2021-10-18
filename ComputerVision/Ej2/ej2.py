import cv2
import numpy as np
import matplotlib.pyplot as plt

class Camera:
    def __init__(self, r, t):
        self.R=np.linalg.inv(r)
        aux1=[-i for i in self.R]
        aux2=[[i] for i in t]
        self.t=np.matmul(aux1, aux2)
        self.comp=np.hstack((self.R, self.t))
    def getT(self):
        return self.t
    def getR(self):
        return self.R
    def setPmatrix(self, K):
        self.P=np.matmul(K, self.comp)
    def getP(self):
        return self.P

class Plane:
    def __init__(self, a,b,c, d):
        self.a=a
        self.b=b
        self.c=c
        self.d=d
    def getNormal(self):
        return np.array([self.a,self.b,self.c])
    def homography(self, k, c1, c2):
        nT=[ self.getNormal()]
        print(nT)
        tn=(np.matmul(c2.getT(), nT))
        tnd=tn/self.d
        corchete=np.subtract(c2.getR(),tnd)
        self.H=np.matmul(np.matmul(k, corchete), np.linalg.inv(k))
        print(self.H)

        
def main():
    r1=np.loadtxt("R_WC1.txt")
    r2=np.loadtxt("R_WC2.txt")
    t1=np.loadtxt("t_WC1.txt")
    t2=np.loadtxt("t_WC2.txt")
    c1=Camera(r1, t1)
    c2=Camera(r2, t2)
    alfa1=458.654
    alfa2=457.296
    x0=367.215
    y0=248.375
    K=np.array([
        [alfa1, 0, x0],
        [0, alfa2, y0],
        [0, 0, 1]]
    )
    C1=Plane(0.243360266797104, -0.738377227649819, -0.440791457369051, 0.448639879349884)
    C1.homography(K, c1, c2)





if __name__ == '__main__':
    main()