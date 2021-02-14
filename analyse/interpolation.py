
import numpy as np

def interpolate_points(p1, p2, n_steps = 10):
    '''
    uniform interpolation between two points in latent space
    '''
    # interpolate ratios between the points
    ratios = np.linspace(0, 1, num=n_steps)
    # linear interpolate vectors
    vectors = list()
    for ratio in ratios:
        v = (1.0 - ratio) * p1 + ratio * p2
        vectors.append(v)
    return np.asarray(vectors)


