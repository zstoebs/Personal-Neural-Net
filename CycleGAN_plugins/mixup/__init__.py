"""
mixup = version of data augmentation
https://arxiv.org/pdf/1710.09412.pdf

simple, low computational overhead
IN TEST --> known to improve CIFAR and ImageNet, TBD on 3D images
"""

def mixup(A0,A1,B0,B1,alpha=0.001):

    lam = np.random.beta(alpha,alpha)
    
    new_A0, new_A1 = self.__make_compat(A0,A1)
    new_B0, new_B1 = self.__make_compat(B0,B1)
    
    A = lam * new_A0 + (1. - lam) * new_A1
    B = lam * new_B0 + (1. - lam) * new_B1
    
    return A, B

def make_compat(t0,t1):
    
    _,x0,y0,z0 = t0.size()
    _,x1,y1,z1 = t1.size()
    
    x = min([x0,x1])
    y = min([y0,y1])
    z = min([z0,z1])
    
    new_t0 = t0[:,:x,:y,:z]
    new_t1 = t1[:,:x,:y,:z]
    
    return new_t0, new_t1
