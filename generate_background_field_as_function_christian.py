""" This is based on the matlab code generate?background_field.m" from Christian Langkammer
The idea is to add this backgroundfield to the rectangles
"""
import numpy as np 
from random import randint
import nibabel as nib
from numpy import matlib
import matplotlib.pyplot as plt
import scipy.ndimage
from itertools import chain
from numpy import array
from visualize_volumes import view_slices_3dNew


def generate_backgroundfield():
     
    
    ###############################################################################
    #load and view phantom 
    phantom = nib.load('christian_files/wisnieff_liu_phantom.nii.gz').get_fdata()
    #view_slices_3dNew(phantom, 100,100,50, vmin=-0.05, vmax=0.05, title='Phantom')
    
    #load and view mask
    mask = nib.load('christian_files/wisnieff_liu_phantom_mask.nii.gz').get_fdata()
    #view_slices_3dNew(mask, 100,100,50, vmin=-0.2, vmax=1.1, title='Phantom Mask')
    
    #load and view phantom background
    phantom_bg = nib.load('christian_files/wisnieff_liu_phantom_background.nii.gz').get_fdata()
    #view_slices_3dNew(phantom_bg, 100,100,50, vmin=-0.05, vmax=0.05, title='Phantom Background')
    ##############################################################################
    
    
    
    #################################################################################
    ################################################################################
    ###############################
    
    ##################################
    #def backgroundfield
    ##################################
    dim = 128
    
    # create new volume
    susc_background = np.zeros((dim, dim, dim))
    
    #% add anterior-posterior susceptibility gradients in the basal gangila
    max_gradient = 0.09#0.01;
    ##############
    
    ### Mask of the phantom 
    mask_bg = phantom > 0.06 #??? False and true  size 256 256 98
    #view_slices_3dNew(mask_bg, 100,100,50, vmin=-0.05, vmax=0.05, title='mask  background   Phantom > 0.06 FILE 256 256 98')
    
    #view_slices_3dNew(mask_bg, 135,37,91, vmin=-0.05, vmax=0.05, title='mask  background   Phantom > 0.06 FILE 256 256 98')
    #view_slices_3dNew(mask_bg, 100 ,37,91, vmin=-0.05, vmax=0.05, title='mask  background  Phantom > 0.06  FILE 256 256 98')
    
    
    ################################################################################
    
    
    #############################################################################
    #############################################################################
    #the mask_bg  has to be converted to 128 128 128
    
    #reduce the first two dimensions
    #take every second element, 256*256*98 --> 128*128*98
    mask_bg_dimNew = mask_bg[0::2, 0::2, : ]#dim 128 128 98
    #plot outcome
    #view_slices_3dNew(mask_bg_dimNew, 67,37,91, vmin=-0.05, vmax=0.05, title='mask  background NEW dimension 128 128 98')
    
    #increase the last
    mask_bg_dimFinal = np.zeros((dim,dim,dim))
    mask_bg_dimFinal[:,:,15:113]= mask_bg_dimNew
    
    #view_slices_3dNew(mask_bg_dimFinal, 67,37,91, vmin=-0.05, vmax=0.05, title='mask  background  Final  128 128 128')
    #view_slices_3dNew(mask_bg_dimFinal, 50,64,60, vmin=-0.05, vmax=0.05, title='Mask Background Final 128 128 128')
    #view_slices_3dNew(mask_bg_dimFinal, 60,64,60, vmin=-0.05, vmax=0.05, title='Mask Background Final 128 128 128')
    
    
    ########################
    #from scipy.ndimage import zoom
    #orig_size = mask_bg.shape
    #mask_bg_dimnewZoom = zoom(mask_bg, (dim / orig_size[0],dim / orig_size[1],dim / orig_size[2]))
    
    #view_slices_3dNew(mask_bg_dimnewZoom, 67,37,91, vmin=-0.05, vmax=0.05, title='mask  background NEW METHOD')
    
    ##############################################################################################
    #####################################################################################
    x = np.linspace(1, dim, dim) #0 to 127
    y = np.linspace(1, dim, dim) #0 to 127
    
    [X,Y] = np.meshgrid(x, y)                         
                                
    
    susc_gradient = X / np.max(X)  * 2 * max_gradient - max_gradient
    
    susc_gradient_max = susc_gradient.max()
    # indexing with np.newaxis inserts a new 3rd dimension, which we then repeat the
    # array along, (you can achieve the same effect by indexing with None, see below)
    susc_gradient3D = np.repeat(susc_gradient[:, :, np.newaxis], 128, axis=2) #128 128 128  
    
    
    
    susc_gradientFinal = np.multiply( susc_gradient3D, mask_bg_dimFinal);
    
    #############
    #sanity check
    susc_gradient3D_max = susc_gradient3D.max()
    susc_gradientFinal_max = susc_gradientFinal.max()
    #find indices where the gradient is not zero!!!!
    susc_grad_indices = np.transpose(np.asarray(np.where(susc_gradientFinal)))
    
    
    
    #view_slices_3dNew(susc_gradientFinal, 50,64,60, vmin=-0.05, vmax=0.05, title='SUSCEPTIBILITY GRADIENT')
    #view_slices_3dNew(susc_gradientFinal, 60,64,60, vmin=-0.05, vmax=0.05, title='SUSCEPTIBILITY GRADIENT')
    
    
    #############################
    
    ########################
    # for plotting
    v_max= 1
    v_min = -1
    ########################
    
    ###############################################################################
    # artifical susc inclusion mimicking air/tissue interfaces 
    ###############################################################################
    #ra = 7;
    r_air = 4
    
    #xx = 135; yy = 180; zz = 20;
    xx = 68; yy = 90; zz = 26
    susc_background[xx-r_air: xx+r_air, 
                    yy-r_air :yy+r_air,
                    zz - r_air: zz + r_air] = 1
    
    tit_air = 'susc Air'
    #view_slices_3dNew(susc_background,
    #               xx, yy, zz,
    #               v_min, v_max,
    #               title=tit_air)
    
    
    ###############################################################################
    # Add Ears
    ###############################################################################
    ######
    #ear radius
    #ra = 5;
    r_ear = 3
    
    
    #######
    #ear1
    #######
    #xx = 24; yy = 97; zz = 54;           24/256*128  97/256*128,  (54/98*128), 
    ear1_cx = 12; ear1_cy = 49; ear1_cz = 70
    susc_background[ear1_cx-r_ear: ear1_cx+r_ear, 
                    ear1_cy-r_ear: ear1_cy+r_ear, 
                    ear1_cz-r_ear: ear1_cz+r_ear] = 1
    
    #####
    #ear2
    #####
    #xx = 230; yy = 97; zz = 54;
    ear2_cx = 115; ear2_cy = 49; ear2_cz = 70
    susc_background[ear2_cx-r_ear: ear2_cx+r_ear, 
                    ear2_cy-r_ear: ear2_cy+r_ear, 
                    ear2_cz-r_ear: ear2_cz+r_ear] = 1
    ###############################################################################
    ###########
    #plots_ears
    ############
    
    
    #view_slices_3dNew(susc_background,
    #               ear1_cx, ear1_cy,ear1_cz,
    #               v_min, v_max,
    #               title='Ears 1')
    
    ################
    #view_slices_3dNew(susc_background,
    #               ear2_cx, ear2_cy,ear2_cz,
    #               v_min, v_max,
    #               title='Ears 2')
    
    ###################################################################
    ######################
    #Add some random fields
    ######################
    #ra = 5
    r_rf= 3
    ###############
    #random field 1
    ###############
    #xx = 39; yy = 204 ; zz = 80
    rf1_cx = 20; rf1_cy = 102; rf1_cz = 104
    
    susc_background[rf1_cx-r_rf: rf1_cx+r_rf, 
                    rf1_cy-r_rf: rf1_cy+r_rf, 
                    rf1_cz-r_rf: rf1_cz+r_rf] = 1
    
    ###############
    #random field 2
    ################
    #xx = 209; yy = 39; zz = 6
    rf2_cx = 105; rf2_cy = 20; rf2_cz = 8
    susc_background[rf2_cx-r_rf: rf2_cx+r_rf, 
                    rf2_cy-r_rf: rf2_cy+r_rf, 
                    rf2_cz-r_rf: rf2_cz+r_rf] = 1
    
    ####################################################################
    ###################
    #plot random fields
    ###################
    '''
    view_slices_3dNew(susc_background,
                   rf1_cx, rf1_cy,rf1_cz,
                   v_min, v_max,
                   title='random field 1')
    
    
    view_slices_3dNew(susc_background,
                   rf2_cx, rf2_cy,rf2_cz,
                   v_min, v_max,
                   title='random field 2')
    '''
    ###############################################################################
    ###############################################################################
    ###################
    # Add all
    ###################
    
    susc_Total = susc_gradientFinal + susc_background
    
    
    
    #view_slices_3dNew(susc_Total, 50,64,48, vmin=-0.05, vmax=0.05, title='SUSCEPTIBILITY GRADIENT')
    
    
    #view_slices_3dNew(susc_Total, 60,64,60, vmin=-0.05, vmax=0.05, title='Mask Background Final 128 128 128')
    
    #view_slices_3dNew(susc_Total, 105,19,60, vmin=-0.05, vmax=0.05, title='Mask Background Final 128 128 128')
    
    return susc_Total
    
background_fied = generate_backgroundfield() #3D 128 128 128 # this is a constant
view_slices_3dNew(background_fied, 50,50,50, vmin=-1, vmax=1, title= "Background ")
view_slices_3dNew(background_fied, 60,64,60, vmin=-0.05, vmax=0.05, title= "Background ")