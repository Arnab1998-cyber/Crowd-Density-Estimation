import torch
import torch.nn as nn


class MCNN(nn.Module):

    def __init__(self):
        super(MCNN,self).__init__()

        self.branch1=nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4)),
            nn.BatchNorm2d(16, eps=0.001, momentum=0, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            nn.Conv2d(16, 32, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)),
            nn.BatchNorm2d(32, eps=0.001, momentum=0, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            nn.Conv2d(32, 16, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)),
            nn.BatchNorm2d(16, eps=0.001, momentum=0, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 8, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)),
            nn.BatchNorm2d(8, eps=0.001, momentum=0, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        self.branch2=nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)),
            nn.BatchNorm2d(20, eps=0.001, momentum=0, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            nn.Conv2d(20, 40, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(40, eps=0.001, momentum=0, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            nn.Conv2d(40, 20, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(20, eps=0.001, momentum=0, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(20, 10, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True)
        )

        self.branch3=nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(24, eps=0.001, momentum=0, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            nn.Conv2d(24, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(48, eps=0.001, momentum=0, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            nn.Conv2d(48, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(24, eps=0.001, momentum=0, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(24, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(12, eps=0.001, momentum=0, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        self.fuse=nn.Sequential(
            nn.Sequential(nn.Conv2d(30, 1, kernel_size=(1, 1), stride=(1, 1))),
            nn.BatchNorm2d(1, eps=0.001, momentum=0, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
    
    def forward(self,img_tensor):
        x1=self.branch1(img_tensor)
        x2=self.branch2(img_tensor)
        x3=self.branch3(img_tensor)
        x=torch.cat((x1,x2,x3),1)
        x=self.fuse(x)
        return x


