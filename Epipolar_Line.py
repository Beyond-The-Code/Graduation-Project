from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import Interest_Point_using_SIFT as Interest_Point
import Matching_using_SIFT as matching
import Essential_Matrix

def compute_epipole(F):
    '''
    Computes the epipole from F for the right epipole.
    Use with F.T for left epipole.
    '''
    # return null space of F (Fx=0)
    U,S,V = np.linalg.svd(F)
    e = V[-1]
    return e/e[2]

def plot_epipolar_line(im,F,x,epipole=None,show_epipole=True):
    '''
    Plot the epipole and epipolar line FX=0 in an image.
    F is the fundamental matrix
    x is point in the other image
    '''
    m,n = im.shape[:2]
    line = np.dot(F,x)
    
    # epipolar line parameter and values
    t = np.linspace(0,n,100)
    lt = np. array([(line[2]+line[0]*tt)/(-line[1]) for tt in t])
    
    # take only line points inside the image
    ndx = (lt>=0) & (lt<m)
    plt.plot(t[ndx],lt[ndx],linewidth=2)
    
    if show_epipole:
        if epipole is None:
            epipole = compute_epipole(F)
    plt.plot(epipole[0]/epipole[2],epipole[1]/epipole[2],"r*")

def plotting(numbers_of_epipolar_line):
    
    # plot each line individually
    for i in range(numbers_of_epipolar_line):
        plot_epipolar_line(im2,F,x1[:,i],E,False)
    plt.axis("off")
    plt.imshow(im1)
    plt.figure()
    
    # plot each point individually
    for i in range(numbers_of_epipolar_line):
        plt.plot(x1[0,i],x1[1,i],"o")
    plt.axis("off")
    # plt.show()
    plt.imshow(im2)
    plt.figure()
    

if __name__=='__main__':   
    
    # calibration
    K = np.array([[2394,0,932],[0,2398,628],[0,0,1]])
    
    # pair of images
    imname1 = "images/1.ppm"
    imname2 = "images/2.ppm"
    
    # convert images into gray scale then make it as array
    im1 = np.array(Image.open(imname1).convert("L"))
    im2 = np.array(Image.open(imname2).convert("L"))
    
    # Process an image and save the results in a file.
    Interest_Point.process_image(imname1,"1.sift")
    Interest_Point.process_image(imname2,"2.sift")
    
    # Read features and return (locations, descriptors).
    l1,d1 = Interest_Point.read_features_from_file("1.sift")
    l2,d2 = Interest_Point.read_features_from_file("2.sift")
    
    # Use match() or match_twosided() function to get matching points between images
    matches = matching.match(d1,d2)
    # matches = matching.match_twosided(d1,d2)
    
    # Match features
    ndx = matches.nonzero()[0]
    ndx2 = [int(matches[i]) for i in ndx]
    
    # Make homogeneous
    x1 = Essential_Matrix.make_homog(l1[ndx,:2].T)
    x2 = Essential_Matrix.make_homog(l2[ndx2,:2].T)
    
    # normalize with inv(K)
    x1n = np.dot(np.linalg.inv(K),x1)
    x2n = np.dot(np.linalg.inv(K),x2)
    
    # Estimate F
    F = Essential_Matrix.compute_fundamental(x1,x2)
    
    E = compute_epipole(F)
    
    # im3 = matching.appendimages(im1,im2)
    
    # plt.imshow(im3)
    
    # numbers_of_epipolar_line=10 
    plotting(10)
