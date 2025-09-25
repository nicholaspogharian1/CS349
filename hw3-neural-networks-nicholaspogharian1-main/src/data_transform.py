import numpy as np


def custom_transform(data):
    """
    Transform the `spiral.csv` data such that it can be more easily classified.

    To pass test_custom_transform_hard, your transformation should create at
    most three features and should allow a LogisticRegression model to achieve
    at least 90% accuracy.

    You can use free_response.q2.visualize_spiral() to visualize the spiral
    as we give it to you, and free_response.q2.visualize_transform() to
    visualize the 3D data transformation you implement here.

    Args:
        data: a Nx2 matrix from the `spiral.csv` dataset.

    Returns:
        A transformed data matrix that is (more) easily classified.
    """
    def spiralfn(alphas, ts):
        #helper function to create plots of spirals
        xcoords = alphas[0]*ts*np.cos(ts)
        xcoordsold = np.copy(xcoords)
        xcoords = xcoords.reshape(xcoords.shape[0],1)
        ycoords = alphas[1]*ts*np.sin(ts)
        ycoordsold = np.copy(ycoords)
        ycoords = ycoords.reshape(ycoords.shape[0],1)
        ret = np.concatenate((xcoords, ycoords), axis=1)
        return ret
    
    spiral0 = spiralfn([1,1],np.linspace(0,20,1000))
    spiral1 = spiralfn([-1,-1],np.linspace(0,20,1000))
    features = np.zeros((data.shape[0], 2))
    
    for i in range(data.shape[0]):
        features[i,0] = (np.ndarray.min(np.apply_along_axis(np.linalg.norm, 1, spiral0-data[i])))
        features[i,1] = (np.ndarray.min(np.apply_along_axis(np.linalg.norm, 1, spiral1-data[i])))
        
    return features
    
