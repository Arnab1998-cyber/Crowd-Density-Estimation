from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import os
import torch
from tqdm import tqdm
from matplotlib import cm as CM
from src.utils.all_utils import read_yaml, log
from src.create_dataset import crowd_data
from src.model import MCNN
import argparse


class training:
    def __init__(self, config_path, image_root, dmap_root):
        content = read_yaml(config_path)

        log_dir = content['base']['log_dir']
        log_filename = content['base']['log_file']
        self.logfile = os.path.join('src', log_dir, log_filename)

        self.shuffle = content['base']['shuffle']
        self.batch_size = content['base']['batch_size']
        self.downsample = content['base']['down_sample']
        self.img_root = image_root
        self.dmap_root = dmap_root

        self.dataset = crowd_data(self.img_root, self.dmap_root, self.downsample)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        self.model = MCNN()
        self.criterian = nn.MSELoss()

        self.cuda = torch.cuda.is_available()

    def fit(self, epochs, learning_rate, momentum, checkpoint_dir, weight_decay=None):
        if len(os.listdir(checkpoint_dir)) > 0:
            checkpoint_path = os.path.join(checkpoint_dir, os.listdir(checkpoint_dir)[0])
            if self.cuda:
                checkpoint = torch.load(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)
        for epoch in range(epochs):
            running_loss = 0
            with tqdm(self.dataloader) as tqdm_epoch:
                for image, label in tqdm_epoch:
                    tqdm_epoch.set_description(f"EPOCH {epoch+1}/{epochs}")
                    
                    image = image/255.0
                    #image = image.cuda()
                    label = label * 100
                    #label = label.cuda()

                    output = self.model(image)
                    loss = self.criterian(output, label)
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    #clip_grad_norm_(mcnn1.parameters(), max_norm=2.0)
                    self.optimizer.step()
                    
                    running_loss += loss.item()
                    tqdm_epoch.set_postfix(loss=loss.item())
                    log('EPOCH {}/{} finished'.format(epoch+1, epochs), self.logfile)
                print('LOSS: ', running_loss/len(self.dataloader))

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            }, 'weights.pt')
            

if __name__ == '__main__':
    config_path = os.path.join('src','config','config.yaml')
    content = read_yaml(config_path)
    log_dir = content['base']['log_dir']
    log_filename = content['base']['log_file']
    logfile = os.path.join('src', log_dir, log_filename)

    image_root = os.path.join('datasets','part_A','train_data','images')
    dmap_root = os.path.join('datasets','part_A','train_data','density_map')

    epochs = content['base']['epoch']
    learning_rate = content['base']['learning_rate']
    momentum = content['base']['momentum']
    checkpoint_dir = os.path.join('src', content['base']['checkpoint'])

    args = argparse.ArgumentParser()
    args.add_argument('--config', '--c', default = config_path)
    args.add_argument('--image', '--i', default = image_root)
    args.add_argument('--dmap', '--d', default = dmap_root)

    args.add_argument('--epoch', '--e', default = epochs)
    args.add_argument('--learning_rate', '--r', default = learning_rate)
    args.add_argument('--momentum', '--m', default = momentum)
    args.add_argument('--checkpoint', '--p', default = checkpoint_dir)
    parsed_args = args.parse_args()

    app = training(parsed_args.config, parsed_args.image, parsed_args.dmap)
    log('Training starting', logfile)
    app.fit(parsed_args.epoch, parsed_args.learning_rate, parsed_args.momentum, parsed_args.checkpoint)
    log('Trianing finished', logfile)

