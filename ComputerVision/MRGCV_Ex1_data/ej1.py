import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

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
    

def main():
    r1=np.loadtxt("R_w_c1.txt")
    r2=np.loadtxt("R_w_c2.txt")
    t1=np.loadtxt("t_w_c1.txt")
    t2=np.loadtxt("t_w_c2.txt")
    c1=Camera(r1, t1)
    c2=Camera(r2, t2)
    alfa1=1165.723022
    alfa2=1165.738037
    x0=649.094971
    y0=484.765015
    K=np.array([
        [alfa1, 0, x0],
        [0, alfa2, y0],
        [0, 0, 1]]
    )
    c1.setPmatrix(K)
    c2.setPmatrix(K)
    p1=c1.getP()
    p2=c2.getP()
    # print(p1)
    # print(p2)

    AP=[3.44,0.8,0.825,1]
    B=[4.2,0.8,0.815, 1]
    C=[4.2, 0.6, 0.820, 1]
    D=[3.55,0.6,0.820, 1]
    E=[-0.01, 2.6, 1.21, 1]
    lis=[AP, B, C, D, E]
    pts1=[]
    pts2=[]
    for i in lis:
        aux=np.matmul(p1, i)
        aux1=[aux[0], aux[1]]
        aux1=[i/aux[2] for i in aux1]
        pts1.append(aux1)
        aux=np.matmul(p2, i)
        aux1=[aux[0], aux[1]]
        aux1=[i/aux[2] for i in aux1]
        pts2.append(aux1)
    pts1=np.array(pts1)
    pts2=np.array(pts2)







    AB=[(AP[i]-B[i]) for i in range(4)]
    print(AB)
    p_ab=np.matmul(p1, AB)
    aux=[p_ab[0], p_ab[1]]
    p_ab_inf=[i/p_ab[2] for i in aux]






    image1 = plt.imread("Image1.jpg")
    plt.imshow(image1)
    plt.scatter(pts1[:, 0], pts1[:, 1], marker="x", color="red", s=200)
    points = [pts1[0],pts1[1]]
    x_coords, y_coords = zip(*points)
    A = np.vstack([x_coords,np.ones(len(x_coords))]).T
    m1, c1 = np.linalg.lstsq(A, y_coords)[0]
    print("Line Solution is y = {m}x + {c}".format(m=m1,c=c1))
    points = [pts1[2],pts1[3]]
    x_coords, y_coords = zip(*points)
    A = np.vstack([x_coords,np.ones(len(x_coords))]).T
    m2, c2 = np.linalg.lstsq(A, y_coords)[0]
    print("Line Solution is y = {m}x + {c}".format(m=m2,c=c2))
    xi = (c1 - c2) / (m2 - m1)
    yi = m1 * xi + c1
    print("INTERSECCION",xi, yi)
    plt.scatter(xi, yi, marker="x", color="black", s=200)
    plt.axline(pts1[0], pts1[1],color="red")
    plt.axline(pts1[2], pts1[3],color="yellow")

    plt.scatter(p_ab_inf[0], p_ab_inf[1], marker="X", color="blue", s=200)
    plt.show()

    p_ab=np.matmul(p2, AB)
    aux=[p_ab[0], p_ab[1]]
    p_ab_inf=[i/p_ab[2] for i in aux]


    image2 = plt.imread("Image2.jpg")
    plt.imshow(image2)
    plt.scatter(pts2[:, 0], pts2[:, 1], marker="x", color="red", s=200)
    points = [pts2[0],pts2[1]]
    x_coords, y_coords = zip(*points)
    A = np.vstack([x_coords,np.ones(len(x_coords))]).T
    m1, c1= np.linalg.lstsq(A, y_coords)[0]
    print("Line Solution is y = {m}x + {c}".format(m=m1,c=c1))
    points = [pts2[2],pts2[3]]
    x_coords, y_coords = zip(*points)
    A = np.vstack([x_coords,np.ones(len(x_coords))]).T
    m2, c2 = np.linalg.lstsq(A, y_coords)[0]
    print("Line Solution is y = {m}x + {c}".format(m=m2,c=c2))
    xi = (c1 - c2) / (m2 - m1)
    yi = m1 * xi + c1
    print("INTERSECCION",xi, yi)
    plt.scatter(xi, yi, marker="x", color="black", s=200)
    plt.axline(pts2[0], pts2[1],color="red")
    plt.axline(pts2[2], pts2[3],color="yellow")
    plt.scatter(p_ab_inf[0], p_ab_inf[1], marker="X", color="blue", s=200)
    plt.show()



    p1=np.array([3.44,0.8,0.825])
    p2=np.array([4.2,0.8,0.815])
    p3=np.array([4.2, 0.6, 0.820])
    p4=np.array([3.55,0.6,0.820])

    pts=[AP, B, C, D]

    U,S,V=np.linalg.svd(pts)
    print('The equation is {0}x + {1}y + {2}z = {3}'.format(V[3][0], V[3][1], V[3][2], V[3][3]))


    dA=(V[3][0]*AP[0]+V[3][1]*AP[1]+V[3][2]*AP[2]+V[3][3])/math.sqrt(V[3][0]*V[3][0]+V[3][1]*V[3][1]+V[3][2]*V[3][2])
    dB=(V[3][0]*B[0]+V[3][1]*B[1]+V[3][2]*B[2]+V[3][3])/math.sqrt(V[3][0]*V[3][0]+V[3][1]*V[3][1]+V[3][2]*V[3][2])
    dC=(V[3][0]*C[0]+V[3][1]*C[1]+V[3][2]*C[2]+V[3][3])/math.sqrt(V[3][0]*V[3][0]+V[3][1]*V[3][1]+V[3][2]*V[3][2])
    dD=(V[3][0]*D[0]+V[3][1]*D[1]+V[3][2]*D[2]+V[3][3])/math.sqrt(V[3][0]*V[3][0]+V[3][1]*V[3][1]+V[3][2]*V[3][2])
    dE=(V[3][0]*E[0]+V[3][1]*E[1]+V[3][2]*E[2]+V[3][3])/math.sqrt(V[3][0]*V[3][0]+V[3][1]*V[3][1]+V[3][2]*V[3][2])
    print("Distancia A",dA,"Distancia B", dB,"Distancia C", dC,"Distancia D", dD,"Distancia E", dE)






if __name__ == '__main__':
    main()