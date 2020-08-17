"""
Functions for applying MRI to CycleGAN
"""

import torch
import torch.nn.functional as F
import math
import nibabel as nib
import numpy as np
import math

def save_mri(image_numpy, image_path, affine=np.eye(4)):
    """Save a numpy image to the disk

    Parameters:
    image_numpy (numpy array) -- input numpy array
    image_path (str)          -- the path of the image
    """
    Imagenii = nib.Nifti1Image(image_numpy,affine,nib.Nifti1Header())
    nib.save(Imagenii, image_path)


def __slice(img, x_pad=256, y_pad=256):
    """
    Slices MRI along the z-dimension and zero pads the 2D slices to 256x256
    """

    # slicing and squeezing (aka removing dims of size 1)
    slices = []
    for slice in torch.split(img,1,dim=-1):
    
        _, _, x, y, _ = list(slice.size())
        assert x <= x_pad and y <= y_pad
        
        unpad = slice.view(1,1,x,y)
        
        x_half = (x_pad - x) / 2
        y_half = (y_pad - y) / 2
        
        padded = F.pad(unpad,pad=(math.floor(y_half),math.ceil(y_half),math.floor(x_half),math.ceil(x_half)),mode='constant',value=0) # zero pad to 256
        
        try:
            assert padded.size()[-2] == x_pad and padded.size()[-1] == y_pad
        except AssertionError:
            print("Slice not zero padded to {0}x{1}: ".format(x_pad,y_pad), padded.size())
        
        slices += [padded]
        
    return slices

# 2D slices for MRI
def get_2D_slices(opt, A, B):
    
    ### 2D SLICING
    A_slices, B_slices = __slice(A), __slice(B)

    num_slices = min(len(A_slices),len(B_slices)) #
    A_stack = torch.cat(A_slices[:num_slices])
    B_stack = torch.cat(B_slices[:num_slices])
    
    # set batch size s.t. bsize != 1 and last batch doesn't have size 1
    bsize = 2
    while num_slices % bsize == 1:
        bsize += 1
    
    A_batches = torch.split(A_stack,bsize,dim=0)
    B_batches = torch.split(A_stack,bsize,dim=0)
    
    return A_batches, B_batches, A_stack, B_stack
