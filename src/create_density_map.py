import numpy as np
import os
import glob
import scipy
import argparse
import scipy.io as io
from src.utils.all_utils import read_yaml, log
import matplotlib.pyplot as plt 
from density_map import density


class genarate_density_map:
    def __init__(self, config_path):
        content = read_yaml(config_path)

        self.leafsize = content['base']['kdtree_leafsize']
        self.nearest_points = content['base']['number_of_nearset_points']

        log_dir = content['base']['log_dir']
        log_filename = content['base']['log_file']
        self.logfile = os.path.join('src', log_dir, log_filename)

        self.density = density(config_path)

    def collect_image_path(self, image_directory):
        img_paths = []
        for img_path in glob.glob(os.path.join(image_directory, '*.jpg')):
            img_paths.append(img_path)
        return img_paths

    def collect_gt_paths(self, gt_directory):
        gt_paths = []
        for gt_path in glob.glob(os.path.join(gt_directory, '*.mat')):
            gt_paths.append(gt_path)
        return gt_paths
        
    def get_density_map(self, img_paths):
        for idx, file in enumerate(img_paths):
            mat = io.loadmat(file.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
            img= plt.imread(file)
            k = np.zeros((img.shape[0],img.shape[1]))
            points = mat["image_info"][0,0][0,0][0]
            map = self.density.density_map(img, points)
            np.save(file.replace('.jpg','.npy').replace('images','ground_truth'), map)
            msg = 'density map of {} image file has been created'.format(idx+1)
            log(msg, self.logfile)
            print(idx+1, 'done')

    
if __name__ == '__main__':
    config_path = os.path.join('src','config','config.yaml')
    content = read_yaml(config_path)
    log_dir = content['base']['log_dir']
    log_filename = content['base']['log_file']
    logfile = os.path.join('src', log_dir, log_filename)

    part_a_test = os.path.join('datasets','part_A','train_data')
    test_image_paths = os.path.join(part_a_test,'images')
    test_gt_paths = os.path.join(part_a_test,'ground-truth')
    
    args = argparse.ArgumentParser()
    args.add_argument('--config', '--c', default = config_path)
    args.add_argument('--image', '--t', default = test_image_paths)
    args.add_argument('--gt', '--g', default = test_gt_paths)
    parsed_args = args.parse_args()

    try:
        app = genarate_density_map(config_path=parsed_args.config)

        log('Start collecting image for {}'.format(parsed_args.image), logfile)
        img_paths = app.collect_image_path(parsed_args.image)
        log('Finish collecting image for {}'.format(parsed_args.image), logfile)

        log('Start collecting ground truth for {}'.format(parsed_args.gt), logfile)
        gt_paths = app.collect_gt_paths(parsed_args.gt)
        log('Finish collecting ground truth for {}'.format(parsed_args.gt), logfile)

        log('Start collection density map for {}'.format(parsed_args.image), logfile)
        app.get_density_map(img_paths)
        log('Finish collection density map for {}'.format(parsed_args.image), logfile)

    except Exception as e:
        print(e)
        log(e, logfile)
        raise e
    