"""
mri_recon
For reconstructing MRI from repeated updates of slices
"""

import torch
import numpy as np
import math

class MRI_Reconstructor:

    def __init__(self):
        """
        Defines a class to process the output of CycleGAN back into MRI
        Behavior:
        - keeps track of visuals in a list
        - once finished, call reconstruct() to piece the output slices / batches back into MRI
        """
        
        self.fake_A_lst = []
        self.fake_B_lst = []
        self.rec_A_lst = []
        self.rec_B_lst = []
        
        self.A_batches = []
        self.B_batches = []
        
        cos_gam = np.cos(math.pi/2)
        sin_gam = np.sin(math.pi/2)
        self.affine = np.array([[cos_gam,-sin_gam,0,0],
                            [sin_gam,cos_gam,0,0],
                            [0,0,1,0],
                            [0,0,0,1]])
    
    def update_visuals(self,visuals):
        """
        Call this after each mini-batch completes forward pass
        Expects output to be of size (batch_size, num_channels, height, width) e.g. (batch_size, 1, 256, 256)
        """
        
        fake_A_vis = visuals['fake_A']
        fake_B_vis = visuals['fake_B']
        rec_A_vis = visuals['rec_A']
        rec_B_vis = visuals['rec_B']
        
#        print("Fake A: ", fake_A_vis.size())
#        print("Fake B: ", fake_B_vis.size())
#        print("Rec A: ", rec_A_vis.size())
#        print("Rec B: ", rec_B_vis.size())
        
        # extracting output images, splitting into individual slices, remove batch size dim from the front for stacking in __process_list()
        self.fake_A_lst += [fake_A_vis]
        self.fake_B_lst += [fake_B_vis]
        self.rec_A_lst += [rec_A_vis]
        self.rec_B_lst += [rec_B_vis]
    
    def reconstruct(self):
        """
        Call this after all minibatches of one MRI have passed
        Expects that each list is of size (1,num_channels,height,width) e.g. (1,1,256,256)
        """
        
        fake_A = self.__process_list(self.fake_A_lst)
        fake_B = self.__process_list(self.fake_B_lst)
        rec_A = self.__process_list(self.rec_A_lst)
        rec_B = self.__process_list(self.rec_B_lst)
        
        return fake_A, fake_B, rec_A, rec_B, self.affine
    
    def __process_list(self, lst):
        """
        Processes a list of tensors each of size (1,num_channels,height,width) into
        a tensor of size (1,height,width,num_channels,num_slices) aka a 3D MRI
        """
        assert len(lst) > 0
        
        img = torch.cat(lst) # size is (n_slices,n_channels,height_width)
        n_slices, n_channels, height, width = list(img.size())
        
        # reshape and squeeze any size 1 dims
        #img = torch.squeeze(img.view(height,width,n_channels,n_slices))
        
        # detach from computation graph, send to cpu
        return np.squeeze(img.detach().cpu().numpy())
    
    def batches(self,input):
        """
        Keeps track of batches for debugging purposes
        Use with recon_from_batches() to check if reconstruction algo works
        """
    
        A_batch = input['A']
        B_batch = input['B']
        
        self.A_batches += [A_batch]
        self.B_batches += [B_batch]
    
    def recon_from_batches(self):
        """
        Use after tracking batches --> primary objective is to check if
        reconstruction algo is correct
        """
        
        rec_A = self.__process_list(self.A_batches)
        rec_B = self.__process_list(self.B_batches)
        
        return rec_A, rec_B
    
    def get_affine(self):
        """
        Return the affine matrix used for the final images
        """
    
        return self.affine
        
    
    
        
