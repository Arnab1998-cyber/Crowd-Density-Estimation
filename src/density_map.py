from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from src.utils.all_utils import read_yaml, create_directory, log
import numpy as np
import os

class density:
    def __init__(self, config_path):
        content = read_yaml(config_path)
        self.leafsize = content['base']['kdtree_leafsize']
        self.nearest_points = content['base']['number_of_nearset_points']  # 4
        log_dir = content['base']['log_dir']
        log_filename = content['base']['log_file']
        self.logfile = os.path.join('src', log_dir, log_filename)

    def density_map(self, image, points):
        image_shape=[image.shape[0],image.shape[1]]
        density = np.zeros(image_shape, dtype=np.float32)
        gt_count = len(points) # number of people in image

        if gt_count == 0:   # if there is no people
            msg = 'image has no people, so return black image'
            log(msg,self.logfile)
            return density
        
        leafsize = self.leafsize
        # build kdtree
        tree = KDTree(points.copy(), leafsize=leafsize)
        # query kdtree
        distances, _ = tree.query(points, k=self.nearest_points) # find 4 nearest neighbour
        
        for i, pt in enumerate(points):

            pt2d = np.zeros(image_shape, dtype=np.float32)
            if int(pt[1])<image_shape[0] and int(pt[0])<image_shape[1]:
                pt2d[int(pt[1]),int(pt[0])] = 1.
            else:
                continue

            if gt_count > 1:
                sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
            else:
                sigma = np.average(np.array(pt2d.shape))/2./2. # there is one person

                # brighten the non zero points i.e., where people present
            density += gaussian_filter(pt2d, sigma, mode='constant')

            msg = 'density map created'
            log(msg,self.logfile)

        return density


