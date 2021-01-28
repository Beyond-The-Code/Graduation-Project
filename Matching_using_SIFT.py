from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import Interest_Point_using_SIFT as Interest_Point

def match(desc1,desc2):
    '''
    matching a feature in one image to a feature in another image is 
    to use the ratio of the distance to the two closest matching features.
    For each descriptor in the first image, select its match in the second image.
    we uses the angle between descriptor vectors as distance measure
    input: 
        desc1 (descriptors for the first image),
        desc2 (descriptors for second image).
    '''
    # Normalize the vectors
    desc1 = np.array([d/np.linalg.norm(d) for d in desc1]) 
    desc2 = np.array([d/np.linalg.norm(d) for d in desc2])
    # the ratio of the distance default is (0.6) you can increase it for more matching
    dist_ratio = 0.6
    desc1_size = desc1.shape
    matchscores = np.zeros((desc1_size[0],1),"int")
    desc2t = desc2.T 
    for i in range(desc1_size[0]):
        dotprods = np.dot(desc1[i,:],desc2t) 
        dotprods = 0.9999*dotprods
        # inverse cosine and sort, return index for features in second image
        indx = np.argsort(np.arccos(dotprods))
        # check if nearest neighbor has angle less than dist_ratio times 2nd
        if np.arccos(dotprods)[indx[0]] < dist_ratio * np.arccos(dotprods)[indx[1]]:
            matchscores[i] = int(indx[0])
            
    return matchscores

def match_twosided(desc1,desc2):
    
    # Two-sided symmetric version of match().
    matches_12 = match(desc1,desc2)
    matches_21 = match(desc2,desc1)
    ndx_12 = matches_12.nonzero()[0]
    # remove matches that are not symmetric
    for n in ndx_12:
        if matches_21[int(matches_12[n])] != n:
            matches_12[n] = 0
            
    return matches_12

def appendimages(im1,im2):
    '''
    Input: im1, im2 (pair of image)
    Return a new image that appends the two images side-by-side.
    '''
    # select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]
    if rows1 < rows2:
        im1 = np.concatenate((im1,np.zeros((rows2-rows1,im1.shape[1]))),axis=0)
    elif rows1 > rows2:
        im2 = np.concatenate((im2,np.zeros((rows1-rows2,im2.shape[1]))),axis=0)
    
        
    return np.concatenate((im1,im2), axis=1)
    

def plot_matches(im1,im2,locs1,locs2,matchscores,show_below=True):
    '''
    Show figure with lines between matches
    input: 
        im1,im2 (images [arrays]),
        locs1,locs2 (locations),
        matchscores (output from match()),
        show_below (if you want to show images below matches). 
    '''
    # Appends the two images side-by-side
    im3 = appendimages(im1,im2)
    if show_below:
        im3 = np.vstack((im3,im3))
	
    plt.imshow(im3)
    
    cols1 = im1.shape[1]
    for i in range(len(matchscores)):
        if matchscores[i] > 0:
            plt.plot([locs1[i,0], locs2[int(matchscores[i,0]),0]+cols1], [locs1[i,1], locs2[int(matchscores[i,0]),1]], 'c')
    plt.axis('off')

if __name__=='__main__':
    
    # pair of images
    imname1="1.ppm"
    imname2="2.ppm"
    
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
    matches = match(d1,d2)
    # matches=match_twosided(d1,d2)
    
    # Plot matching points
    plt.figure()
    plt.gray()
    plot_matches(im1,im2,l1,l2,matches)
    plt.show()
        