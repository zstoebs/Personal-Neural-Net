import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import nibabel as nib
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from mixup import mixup


### ensure that dataset_mode opt is set to 'mri'
class MRIDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        if self.opt.isTrain:
        
            index_A0 = index % self.A_size
            A0_path = self.A_paths[index_A0]  # make sure index is within then range
            index_A1 = self.__get_rand_idx([index_A0],self.A_size)
            A1_path = self.A_paths[index_A1]
            
            if self.opt.serial_batches:   # make sure index is within then range
                index_B0 = index % self.B_size
            else:   # randomize the index for domain B to avoid fixed pairs.
                index_B0 = random.randint(0, self.B_size - 1)
                
            index_B1 = self.__get_rand_idx([index_B0],self.B_size)
                    
            B0_path = self.B_paths[index_B0]
            B1_path = self.B_paths[index_B1]
            
            devA_idx = self.__get_rand_idx([index_A0,index_A1],self.A_size)
            devA_path = self.A_paths[devA_idx]
            devB_idx = self.__get_rand_idx([index_B0,index_B1],self.B_size)
            devB_path = self.B_paths[devB_idx]
            
            A0_img = self.__load_and_normalize(A0_path)
            A1_img = self.__load_and_normalize(A1_path)
            B0_img = self.__load_and_normalize(B0_path)
            B1_img = self.__load_and_normalize(B1_path)
            devA_img = self.__load_and_normalize(devA_path)
            devB_img = self.__load_and_normalize(devB_path)
            
    #        imA=nib.Nifti1Image(A_img, np.eye(4),nib.Nifti1Header())
    #        nib.save(imA, os.path.join(os.getcwd(),"train_imgs","imA{0}".format(index)))
    #        imB=nib.Nifti1Image(B_img, np.eye(4),nib.Nifti1Header())
    #        nib.save(imB, os.path.join(os.getcwd(),"train_imgs","imB{0}".format(index)))
    #        print("Saved immediate imgs...")

            A0 = torch.FloatTensor([A0_img])
            B0 = torch.FloatTensor([B0_img])
            A1 = torch.FloatTensor([A1_img])
            B1 = torch.FloatTensor([B1_img])
            devA = torch.FloatTensor([devA_img])
            devB = torch.FloatTensor([devB_img])
            
            A, B = mixup(A0,A1,B0,B1)
            
            A_path = [A0_path,A1_path,devA_path]
            B_path = [B0_path,B1_path,devB_path]
        else:
             
            index_A0 = index % self.A_size
            A0_path = self.A_paths[index_A0]
            
            if self.opt.serial_batches:   # make sure index is within then range
                index_B0 = index % self.B_size
                
            else:   # randomize the index for domain B to avoid fixed pairs.
                index_B0 = random.randint(0, self.B_size - 1)

            B0_path = self.B_paths[index_B0]
            
            A0_img = self.__load_and_normalize(A0_path)
            B0_img = self.__load_and_normalize(B0_path)
            
            A0 = torch.FloatTensor([A0_img])
            B0 = torch.FloatTensor([B0_img])
            
            A = A0
            B = B0
            
            devA = []
            devB = []
            
            A_path = A0_path
            B_path = B0_path
        
        # apply random flip to the images
        A = self.__rand_flip(A)
        B = self.__rand_flip(B)
                
        return {'A0':A0,'B0':B0,'A': A, 'B': B, 'devA': devA, 'devB': devB,'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
    
    def __load_and_normalize(self,img_path):
    
        nifty_img = nib.load(img_path)
        img_data = nifty_img.get_fdata()
        ##since the image is nii version, we need to normalize it
        img = (img_data - np.min(img_data))/(np.max(img_data) - np.min(img_data))
        
        return img
    
    def __rand_flip(self,img):
    
        return torch.flip(img,[1,2]) if random.random() >= 0.5 else img
    
    def __get_rand_idx(self, orig_idx: list, size: int):
    
        new_idx = random.randint(0,size - 1)
        while new_idx in orig_idx:
            new_idx = random.randint(0,size - 1)
        return new_idx
