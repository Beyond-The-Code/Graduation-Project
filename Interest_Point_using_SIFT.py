from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from numpy import *
from pylab import *

def process_image(imagename,resultname,params="--edge-thresh 10 --peak-thresh 5"):
    '''
    To compute SIFT features for images we will use the open source package VLFeat
    The binaries need the image in grayscale .pgm format
    Process an image convert image into gray scale and save result into temporary file .pgm format.
    each row contains the coordinates, scale and rotation angle 
    for each interest point as the first four values,
    followed by the 128 values of the corresponding descriptor.'''
    
    if imagename[-3:] != "pgm":
        im = Image.open(imagename).convert("L")
        im.save("tmp.pgm")
        imagename = "tmp.pgm"
        cmmd = str("sift "+imagename+" --output="+resultname+" "+params)
        os.system(cmmd)
        
def read_features_from_file(filename):
    # Read feature properties and return in matrix form (feature locations, descriptors).
    f = np.loadtxt(filename)
    return f[:,:4],f[:,4:]

def read_features_from_csv_file(filename):
    # read feature from csv file
    return np.loadtxt(filename, delimiter=",")

def write_features_to_file(filename,l,d):
    # Save feature location and descriptor to file.
    np.savetxt(filename,hstack((l,d)))

def plot_features(im,l,circle=False):
    ''' 
    Show image with features. 
    input: 
        im (image as array),
        l (row, col, scale, orientation of each feature).
    This will plot the location of the SIFT points as blue dots overlaid on the image.
    If the parameter circle is set to "True", circles with radius equal to the scale of the
    feature will be drawn instead using the helper function draw_circle().
    '''

    def draw_circle(c,r):
        t = arange(0,1.01,.01)*2*pi
        x = r*cos(t) + c[0]
        y = r*sin(t) + c[1]
        plt.plot(x,y,'b',linewidth=2)
    plt.imshow(im)
    if circle:
        for p in l:
            draw_circle(p[:2],p[2])
    else:
            plot(l[:,0],l[:,1],'ob')
    axis('off')

if __name__=='__main__':
    
    imname1 = "images/1.ppm"
    
    # convert images into gray scale then make it as array
    im = np.array(Image.open(imname1).convert("L"))
    
    # Process an image and save the results in a file.
    process_image(imname1,'1.sift')
    
    # Read features and return (locations, descriptors).
    l,d=read_features_from_file('1.sift')
    
    # Plot features of image 
    plt.figure()
    plt.gray()
    plot_features(im,l,circle=True)
    plt.show()
