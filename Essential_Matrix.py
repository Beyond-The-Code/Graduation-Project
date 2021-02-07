from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
import Interest_Point_using_SIFT as Interest_Point
import Matching_using_SIFT as matching

def make_homog(points):
    """ Convert a set of points (dim*n array) to homogeneous coordinates. """
    return np.vstack((points,np.ones((1,points.shape[1]))))

def compute_fundamental(x1,x2):
    '''
    Computes fundamental matrix from corresponding points using the normalized 8 point algorithm.
    each row is [x’*x, x’*y, x’, y’*x, y’*y, y’, x, y, 1]
    '''
    n = x1.shape[1]
    if x2.shape[1] != n:
            raise ValueError("Number of points don’t match.")

    A = np.zeros((n,9))
    for i in range(n):
        A[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
                x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
                x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i]]

    U,S,V = np.linalg.svd(A)
    F = V[-1].reshape(3,3)
    
    # make rank 2 by put zero in last singular value
    U,S,V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U,np.dot(np.diag(S),V))
    return F

def compute_fundamental_normalized(x1,x2):
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don’t match.")
    
    # normalize coordinates
    x1 = x1 / x1[2]
    mean_1 = np.mean(x1[:2],axis=1)
    S1 = math.sqrt(2) / np.std(x1[:2])
    T1 = np.array([[S1,0,-S1*mean_1[0]],[0,S1,-S1*mean_1[1]],[0,0,1]])
    x1 = np.dot(T1,x1)
    x2 = x2 / x2[2]
    mean_2 = np.mean(x2[:2],axis=1)
    S2 = math.sqrt(2) / np.std(x2[:2])
    T2 = np.array([[S2,0,-S2*mean_2[0]],[0,S2,-S2*mean_2[1]],[0,0,1]])
    x2 = np.dot(T2,x2)
    F = compute_fundamental(x1,x2)
    # reverse normalization
    F = np.dot(T1.T,np.dot(F,T2))
    return F/F[2,2]

class RansacModel(object):
    ''' 
    RANSAC is an iterative method to fit models to data that can contain outliers.
    Class for fundmental matrix fit with ransac.py from http://www.scipy.org/Cookbook/RANSAC
    '''
    def __init__(self,debug=False):
        self.debug = debug
    def fit(self,data):
        # compute fundamental matrix using eight correspondences
        
        # transpose and split data into the two point sets
        data = data.T
        x1 = data[:3,:8]
        x2 = data[3:,:8]
        
        F = compute_fundamental_normalized(x1,x2)
        return F
    def get_error(self,data,F):
        '''
        Compute x^T F x for all correspondences, return error for each transformed point.
        '''
        data = data.T
        x1 = data[:3]
        x2 = data[3:]
        
        Fx1 = np.dot(F,x1)
        Fx2 = np.dot(F,x2)
        denom = Fx1[0]**2 + Fx1[1]**2 + Fx2[0]**2 + Fx2[1]**2
        err = ( np.diag(np.dot(x1.T,np.dot(F,x2))) )**2 / denom
        return err

def ransac(data,model,n,k,t,d,debug=False,return_all=False):

    iterations = 0
    bestfit = None
    besterr = np.inf
    best_inlier_idxs = None
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n,data.shape[0])
        maybeinliers = data[maybe_idxs,:]
        test_points = data[test_idxs]
        maybemodel = model.fit(maybeinliers)
        test_err = model.get_error( test_points, maybemodel)
        also_idxs = test_idxs[test_err < t] # select indices of rows with accepted points
        alsoinliers = data[also_idxs,:]
        if debug:
            print ('test_err.min()',test_err.min())
            print ('test_err.max()',test_err.max())
            print ('numpy.mean(test_err)',np.mean(test_err))
            print ('iteration %d:len(alsoinliers) = %d'%(
                iterations,len(alsoinliers)))
        if len(alsoinliers) > d:
            betterdata = np.concatenate( (maybeinliers, alsoinliers) )
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error( betterdata, bettermodel)
            thiserr = np.mean( better_errs )
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate( (maybe_idxs, also_idxs) )
        iterations+=1
    if bestfit is None:
        raise ValueError("did not meet fit acceptance criteria")
    if return_all:
        return bestfit, {'inliers':best_inlier_idxs}
    else:
        return bestfit

def random_partition(n,n_data):
    """return n random rows of data (and also the other len(data)-n rows)"""
    all_idxs = np.arange( n_data )
    np.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2

def F_from_ransac(x1,x2,model,maxiter=5000,match_theshold=1e-6):
    '''
    The function apply threshold and the minimum number of points desired.
    The most important parameter is the maximum number of iterations, 
    exiting too early might give a worse solution, too many iterations will take more time.
    '''
    # import ransac
    data = np.vstack((x1,x2))
    F,ransac_data = ransac(data.T,model,8,maxiter,match_theshold,20,return_all=True)
    return F, ransac_data['inliers']

def compute_P_from_essential(E):
    '''
    Computes second camera matrix P1 = [I 0] from an essential matrix.
    Output: four possible camera matrices.
    '''
    # confirm that E with two equal non-zero singular values (rank is 2)
    U,S,V = np.linalg.svd(E)
    if np.linalg.det(np.dot(U,V))<0:
        V = -V
    # Hartley & zisser man
    E = np.dot(U,np.dot(np.diag([1,1,0]),V))
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    
    # All four solutions
    P2 = [np.vstack((np.dot(U,np.dot(W,V)).T,U[:,2])).T,
          np.vstack((np.dot(U,np.dot(W,V)).T,-U[:,2])).T,
          np.vstack((np.dot(U,np.dot(W.T,V)).T,U[:,2])).T,
          np.vstack((np.dot(U,np.dot(W.T,V)).T,-U[:,2])).T]
    
    for i in range(4):
        P2[i] = np.vstack((P2[i],[0,0,0,1]))
        
    return P2

def Testing_4_Possibilities(x1,x2):
    
    p1x = np.array([[0,-x1[2,0],x1[1,0]],[x1[2,0],0,-x1[0,0]],[-x1[1,0],x1[0,0],0]])   
    p2x = np.array([[0,-x2[2,0],x2[1,0]],[x2[2,0],0,-x2[0,0]],[-x2[1,0],x2[0,0],0]])
    
    M1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])

    for i in range(4):
            Hresult_c1_c2 = np.linalg.inv(Hresult_c2_c1[i])
            M2 = Hresult_c1_c2[0:3,0:4]
            A = np.vstack((np.dot(p1x,M1),np.dot(p2x,M2)))
            U,S,V = np.linalg.svd(A)
            P = V[-1]
            P1est = P/P[3]
            P2est = np.dot(Hresult_c1_c2,P1est)
            if (P1est[2] > 0 and P2est[2]) > 0:
                Hest_c2_c1 = Hresult_c2_c1[i]
                break
    return Hest_c2_c1,P1est

def Reconstructing_Rest_Of_Points(Hest_c2_c1,P1est):
    Hest_c1_c2 = np.linalg.inv(Hest_c2_c1)
    M2est = Hest_c1_c2[0:3,:]
    P1est = P1est.reshape(4,1)
    m,n = x1n.shape[:2]
    M1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    for i in range(1,n):
        p1x = np.array([[0,-x1n[2,i],x1n[1,i]],[x1n[2,i],0,-x1n[0,i]],[-x1n[1,i],x1n[0,i],0]])
        p2x = np.array([[0,-x2n[2,i],x2n[1,i]],[x2n[2,i],0,-x2n[0,i]],[-x2n[1,i],x2n[0,i],0]])    
        A = np.vstack((np.dot(p1x,M1),np.dot(p2x,M2est)))
        U,S,V = np.linalg.svd(A)
        P = V[-1].reshape(4,1)
        P1est = np.hstack((P1est,P/P[3]))
    return P1est

def Plot_Model(P1est):
    ax = plt.axes(projection='3d')
    ax.scatter3D(P1est[0,:],P1est[1,:],P1est[2,:], cmap='Greens');


if __name__=='__main__':   
    
    # calibration
    K = np.array([[2394,0,932],[0,2398,628],[0,0,1]])
    
    # pair of images
    imname1 = "images/1.png"
    imname2 = "images/2.png"
    
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
    
    ndx = matches.nonzero()[0]
    ndx2 = [int(matches[i]) for i in ndx]
    
    x1 = make_homog(l1[ndx,:2].T)
    x2 = make_homog(l2[ndx2,:2].T)
    
    x1n = np.dot(np.linalg.inv(K),x1)
    x2n = np.dot(np.linalg.inv(K),x2)
    
    # Estimate E with RANSAC
    model = RansacModel()
    E,inliers = F_from_ransac(x1n,x2n,model)
    
    # Hresult_c2_c1 will be list of four solutions
    Hresult_c2_c1 = compute_P_from_essential(E)
    
    Hest_c2_c1,P1est = Testing_4_Possibilities(x1n,x2n)
    
    P1est = Reconstructing_Rest_Of_Points(Hest_c2_c1,P1est)
    
    Plot_Model(P1est)
