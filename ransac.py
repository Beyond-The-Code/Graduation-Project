import numpy
import scipy
import scipy.linalg

def ransac(data,model,n,k,t,d,debug=False,return_all=False):

    iterations = 0
    bestfit = None
    besterr = numpy.inf
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
            print ('numpy.mean(test_err)',numpy.mean(test_err))
            print ('iteration %d:len(alsoinliers) = %d'%(
                iterations,len(alsoinliers)))
        if len(alsoinliers) > d:
            betterdata = numpy.concatenate( (maybeinliers, alsoinliers) )
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error( betterdata, bettermodel)
            thiserr = numpy.mean( better_errs )
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = numpy.concatenate( (maybe_idxs, also_idxs) )
        iterations+=1
    if bestfit is None:
        raise ValueError("did not meet fit acceptance criteria")
    if return_all:
        return bestfit, {'inliers':best_inlier_idxs}
    else:
        return bestfit

def random_partition(n,n_data):
    """return n random rows of data (and also the other len(data)-n rows)"""
    all_idxs = numpy.arange( n_data )
    numpy.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2

class LinearLeastSquaresModel:
    """linear system solved using linear least squares

    This class serves as an example that fulfills the model interface
    needed by the ransac() function.
    
    """
    def __init__(self,input_columns,output_columns,debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug
    def fit(self, data):
        A = numpy.vstack([data[:,i] for i in self.input_columns]).T
        B = numpy.vstack([data[:,i] for i in self.output_columns]).T
        x,resids,rank,s = scipy.linalg.lstsq(A,B)
        return x
    def get_error( self, data, model):
        A = numpy.vstack([data[:,i] for i in self.input_columns]).T
        B = numpy.vstack([data[:,i] for i in self.output_columns]).T
        B_fit = scipy.dot(A,model)
        err_per_point = numpy.sum((B-B_fit)**2,axis=1) # sum squared error per row
        return err_per_point
        
def test():
    # generate perfect input data

    n_samples = 500
    n_inputs = 1
    n_outputs = 1
    A_exact = 20*numpy.random.random((n_samples,n_inputs) )
    perfect_fit = 60*numpy.random.normal(size=(n_inputs,n_outputs) ) # the model
    B_exact = scipy.dot(A_exact,perfect_fit)
    assert B_exact.shape == (n_samples,n_outputs)

    # add a little gaussian noise (linear least squares alone should handle this well)
    A_noisy = A_exact + numpy.random.normal(size=A_exact.shape )
    B_noisy = B_exact + numpy.random.normal(size=B_exact.shape )

    if 1:
        # add some outliers
        n_outliers = 100
        all_idxs = numpy.arange( A_noisy.shape[0] )
        numpy.random.shuffle(all_idxs)
        outlier_idxs = all_idxs[:n_outliers]
        non_outlier_idxs = all_idxs[n_outliers:]
        A_noisy[outlier_idxs] =  20*numpy.random.random((n_outliers,n_inputs) )
        B_noisy[outlier_idxs] = 50*numpy.random.normal(size=(n_outliers,n_outputs) )

    # setup model

    all_data = numpy.hstack( (A_noisy,B_noisy) )
    input_columns = range(n_inputs) # the first columns of the array
    output_columns = [n_inputs+i for i in range(n_outputs)] # the last columns of the array
    debug = False
    model = LinearLeastSquaresModel(input_columns,output_columns,debug=debug)

    linear_fit,resids,rank,s = scipy.linalg.lstsq(all_data[:,input_columns],
                                                  all_data[:,output_columns])

    # run RANSAC algorithm
    ransac_fit, ransac_data = ransac(all_data,model,
                                     50, 1000, 7e3, 300, # misc. parameters
                                     debug=debug,return_all=True)
    if 1:
        import pylab

        sort_idxs = numpy.argsort(A_exact[:,0])
        A_col0_sorted = A_exact[sort_idxs] # maintain as rank-2 array

        if 1:
            pylab.plot( A_noisy[:,0], B_noisy[:,0], 'k.', label='data' )
            pylab.plot( A_noisy[ransac_data['inliers'],0], B_noisy[ransac_data['inliers'],0], 'bx', label='RANSAC data' )
        else:
            pylab.plot( A_noisy[non_outlier_idxs,0], B_noisy[non_outlier_idxs,0], 'k.', label='noisy data' )
            pylab.plot( A_noisy[outlier_idxs,0], B_noisy[outlier_idxs,0], 'r.', label='outlier data' )
        pylab.plot( A_col0_sorted[:,0],
                    numpy.dot(A_col0_sorted,ransac_fit)[:,0],
                    label='RANSAC fit' )
        pylab.plot( A_col0_sorted[:,0],
                    numpy.dot(A_col0_sorted,perfect_fit)[:,0],
                    label='exact system' )
        pylab.plot( A_col0_sorted[:,0],
                    numpy.dot(A_col0_sorted,linear_fit)[:,0],
                    label='linear fit' )
        pylab.legend()
        pylab.show()

if __name__=='__main__':
    test()
    
