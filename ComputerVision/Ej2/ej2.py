import cv2
import numpy as np
import matplotlib.pyplot as plt

H1=np.array([])
H2=np.array([])
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
        tn=(np.matmul(c1.getT(), nT))
        tnd=tn/self.d
        corchete=np.subtract(c1.getR(),tnd)
        self.HC1=np.matmul(np.matmul(k, corchete), np.linalg.inv(k))
        tn=(np.matmul(c2.getT(), nT))
        tnd=tn/self.d
        corchete=np.subtract(c2.getR(),tnd)
        self.HC2=np.matmul(np.matmul(k, corchete), np.linalg.inv(k))
        self.HC2C1=np.matmul(self.HC2, np.linalg.inv(self.HC1))
        self.HC1C2=np.matmul(self.HC1, np.linalg.inv(self.HC2))
        return self.HC1C2, self.HC2C1
        
def onclick1(X, Y):
    print("Original: ", X, Y)
    p=np.array([[X],[Y],[1]])
    p2=np.matmul(H2, p)
    punto=[p2[i]/p2[2] for i in range(2)]
    print("Nuevo: ", punto[0], punto[1])
    plt.figure(2)
    plt.scatter(punto[0], punto[1], marker="X", color="red", s=100)
    plt.show()

    

        
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
    global H1, H2
    H1, H2=C1.homography(K, c1, c2)
    print("Homography: \n",H2)
    plt.figure(1)
    image1 = plt.imread("image1_undistorted.png")
    plt.imshow(image1)
    #cid = fig.canvas.mpl_connect('button_press_event', onclick1)
    #coord_clicked_point = plt.ginput(1, show_clicks=False) 
    plt.figure(2)
    image2 = plt.imread("image2_undistorted.png")
    plt.imshow(image2)
    while(True):
        plt.figure(1)
        plt.imshow(image1)
        point = plt.ginput(1, show_clicks=False) 
        if not point:  # List empty
            break
        onclick1(point[0][0], point[0][1])










if __name__ == '__main__':
    main()