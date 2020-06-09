import nibabel as nb
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output
import scipy.ndimage.interpolation as interpolation
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('image', aspect='equal')
import SimpleITK as sitk
import numpy as np
import os


from nipype.interfaces.ants import N4BiasFieldCorrection
from nipype.interfaces.fsl import FLIRT, BET, ApplyMask, ApplyXFM, ConvertXFM, BinaryMaths, ChangeDataType, MultiImageMaths
from nipype.interfaces.ants import N4BiasFieldCorrection


def resampleit(image, dims, isseg=False):
    order = 0 if isseg == True else 5
    
    image = interpolation.zoom(image, np.array(dims)/np.array(image.shape, dtype=np.float32), order=order, mode='nearest')

    if image.shape[-1] == 3: #rgb image
        return image
    else:
        return image 
    
    
    
def func_resample_isovoxel(img_original, isseg=False):
    
    oldimage = sitk.ReadImage(img_original)
    coord = oldimage.GetDirection()
    oldimage_arr = sitk.GetArrayFromImage(oldimage)

    size = oldimage.GetSize()
    spacing = oldimage.GetSpacing()

    fovx = size[0] * spacing[0]
    fovy = size[1] * spacing[1]
    fovz = size[2] * spacing[2]

    dimx = int(round(fovx/1.0))
    dimy = int(round(fovy/1.0))
    dimz = int(round(fovz/1.0))

    newimage_arr = resampleit(oldimage_arr, (dimz, dimy, dimx), isseg=isseg)
    newimage = sitk.GetImageFromArray(newimage_arr)
    newimage.SetDirection(coord)
    return newimage
    
    
    
def func_register(img_original,img_template, img_registered):
    # img_original : original T2 or FLAIR image file
    # img_template : isovoxel T1C file used for registration template
    # img_registered :file name that stores registered (isovoxel) T2 or FLAIR file
    coregi_iso = FLIRT(bins=640, cost_func='mutualinfo', dof=12, output_type="NIFTI_GZ", verbose=0,
                          datatype = 'float', interp = 'trilinear',
                          in_file = img_original, 
                          reference = img_template,
                          out_file = img_registered)
    coregi_iso.run()
                       


def func_n4bias(img_original, img_biascorrected):
    n4_correct = N4BiasFieldCorrection(dimension=3, bspline_fitting_distance=300, bspline_order = 3, 
                                       shrink_factor = 2, n_iterations=[100, 100, 100, 100],
                                       convergence_threshold= 1e-06,
                                       terminal_output = 'none',
                                       input_image = img_original,
                                       output_image = img_biascorrected)

    n4_correct.run()
        

# SI normalization for U-Net segmentation 
def func_norm_unet(img_arr, brain_mask_arr):
    
    img_arr = img_arr * brain_mask_arr   # Assign 0 to background pixels

    mean = np.mean(img_arr[img_arr > 0])
    sd = np.std(img_arr[img_arr > 0])

    img_norm_arr = (img_arr - mean) / sd
    img_norm_arr[img_norm_arr < - 5] = - 5
    img_norm_arr[img_norm_arr >   5] = 5
    img_norm_arr = img_norm_arr / 10
    img_norm_arr = img_norm_arr + 0.5
    img_norm_arr = img_norm_arr * brain_mask_arr  ## Assign 0 to the background
    
    return img_norm_arr
    
    
    
    
# Background crop -> resample to 128 x 128 x 128 
def func_get_cropdown_info(img_arr):
    
    ## finding a 3D bounding box

    x_len,y_len,z_len = img_arr.shape

    for x in range(x_len):
        if np.sum(img_arr[x,:,:]) > 0:
            x_min = x
            break

    for x in range(1, x_len):
        if np.sum(img_arr[(x_len-x),:,:]) > 0:
            x_max = x_len-x
            break

    for y in range(y_len):
        if np.sum(img_arr[:,y,:]) > 0:
            y_min = y
            break

    for y in range(1, y_len):
        if np.sum(img_arr[:,(y_len-y),:]) > 0:
            y_max = y_len-y
            break

    for z in range(z_len):
        if np.sum(img_arr[:,:,z]) > 0:
            z_min = z
            break

    for z in range(1, z_len):
        if np.sum(img_arr[:,:,(z_len-z)]) > 0:
            z_max = z_len-z
            break


    # calculate center location and width of even number
    x_center = int(np.mean((x_min, x_max)))
    x_width = int((x_max - x_min)/2) * 2
    y_center = int(np.mean((y_min, y_max)))
    y_width = int((y_max - y_min)/2) * 2
    width = max(x_width, y_width)   # to make XY plane square, from rectangle

    z_center = int(np.mean((z_min, z_max)))
    z_width = int((z_max - z_min)/2) * 2

    x_min = int(x_center - width/2)
    x_max = int(x_center + width/2)
    y_min = int(y_center - width/2)
    y_max = int(y_center + width/2)

    z_min = int(z_center - z_width/2)
    z_max = int(z_center + z_width/2)

    ## Give ~6 pixels of background margin for xy plane, ~2 pixels for z-axis, within original image border
    img_arr_crop = img_arr[max(0, x_min-6):min(x_len, x_max+6),   
                           max(0, y_min-6):min(y_len, y_max+6),
                           max(0, z_min-2):min(z_len, z_max+2)]

    res_orig = img_arr_crop.shape  # this image is to resampled to 128x128x128.

    return_list=[[res_orig,
                 max(0, x_min-6), min(x_len, x_max+6), x_len,   ## Give some background margin 
                 max(0, y_min-6), min(y_len, y_max+6), y_len,
                 max(0, z_min-2), min(z_len, z_max+2), z_len]]
    return_df = pd.DataFrame(return_list)
    return_df.columns= ('res_orig', 
                        'x_min2','x_max2', 'x_len',
                        'y_min2', 'y_max2','y_len',
                        'z_min2', 'z_max2', 'z_len')


    return return_df


def func_img_cropdown(img_arr, cropdown_info):
    
    x_min2 = cropdown_info.loc[:,'x_min2'].iloc[0]
    x_max2 = cropdown_info.loc[:,'x_max2'].iloc[0]
    y_min2 = cropdown_info.loc[:,'y_min2'].iloc[0]
    y_max2 = cropdown_info.loc[:,'y_max2'].iloc[0]
    z_min2 = cropdown_info.loc[:,'z_min2'].iloc[0]
    z_max2 = cropdown_info.loc[:,'z_max2'].iloc[0]
    
    img_arr_crop = img_arr[x_min2:x_max2,
                            y_min2:y_max2,
                            z_min2:z_max2]
    img_arr_cropdown = resampleit(img_arr_crop, (128,128,128), isseg=False)
    
    return img_arr_cropdown
    
    
    
######################################################################################
def func_img_proc(T1C_bet, T2_bet, FLAIR_bet, mask_T1C_bet,
                  T1C_isovoxel, T2_isovoxel, FLAIR_isovoxel, mask_T1C_bet_iso,
                  T1C_corrected, T2_corrected, FLAIR_corrected):
    
    t1c_isovoxel = func_resample_isovoxel(T1C_bet, isseg=False)
    sitk.WriteImage(t1c_isovoxel, T1C_isovoxel)
    
    bmask_isovoxel = func_resample_isovoxel(mask_T1C_bet, isseg=True)
    sitk.WriteImage(bmask_isovoxel, mask_T1C_bet_iso)
    
    print("resampling T1C & brain mask - completed")
    
    func_register(T2_bet, T1C_isovoxel, T2_isovoxel)
    print("register T2 to T1C_isovoxel - completed")
    
    func_register(FLAIR_bet, T1C_isovoxel, FLAIR_isovoxel)
    print("register FLAIR to T1C_isovoxel - completed")
   
    func_n4bias(T1C_isovoxel, T1C_corrected)
    print("T1C bias correction done...")
    func_n4bias(T2_isovoxel, T2_corrected)
    print("T2 bias correction done...")
    func_n4bias(FLAIR_isovoxel, FLAIR_corrected)
    print("FLAIR bias correction done...")

    
def func_norm_model1(T1C_corrected, FLAIR_corrected, mask_T1C_bet_iso):
    t1c_corrected = nb.load(T1C_corrected)
    t1c_corrected_arr = t1c_corrected.get_data()
    flair_corrected = nb.load(FLAIR_corrected)
    flair_corrected_arr = flair_corrected.get_data()
    brain_mask = nb.load(mask_T1C_bet_iso)
    brain_mask_arr = brain_mask.get_data()
    
    ### normalization for UNet  -> resize to 128 x 128 x 128 
    
    t1c_norm_unet_arr = func_norm_unet(t1c_corrected_arr, brain_mask_arr)
    flair_norm_unet_arr = func_norm_unet(flair_corrected_arr, brain_mask_arr)

    cropdown_info = func_get_cropdown_info(t1c_norm_unet_arr)
    t1c_norm_unet_cropdown_arr = func_img_cropdown(t1c_norm_unet_arr, cropdown_info)
    flair_norm_unet_cropdown_arr = func_img_cropdown(flair_norm_unet_arr, cropdown_info)
   
    
    print("Image SI normalization & resizing :  done...")
   
    return (t1c_norm_unet_cropdown_arr, flair_norm_unet_cropdown_arr, cropdown_info)


    
##########################################################
    
    
    
from UNet3d_architecture import *
    
def func_get_predmask(t1c_arr, flair_arr):
    
    print("Calling pretrained Model 1...")
    model_unet = UNet_n_base(in_channels=2, class_number=2, n_base_filter=21)  
    model_filename = 'MODEL1_UNet_segmentation.pth'
    checkpoint = torch.load(model_filename)
    model_unet.load_state_dict(checkpoint['model_state_dict'])
    model_unet.eval()
    model_unet.cuda()
    
    
    print("Acquiring predicted tumor segmentation mask...")
    t1c_arr = np.expand_dims(t1c_arr, axis = 0)
    flair_arr = np.expand_dims(flair_arr, axis = 0)

    input_arr = np.concatenate([t1c_arr, flair_arr], axis = 0) #shape (2,128,128,128)
    input_arr = np.expand_dims(input_arr, axis = 0) #shape (1, 2,128,128,128)
    
    input_arr = torch.from_numpy(input_arr).float()
    input_arr = input_arr.cuda()
    with torch.no_grad():
        output = model_unet(input_arr)
        output = torch.sigmoid(output)
    output = output.cpu().numpy()
    outputs2 = output[:,0:2,:,:,:]
    outputs2_bin = np.copy(outputs2)
    outputs2_bin[outputs2_bin>0.95] = 1
    outputs2_bin[outputs2_bin<=0.95] = 0
    predmask_arr = outputs2_bin[0,0,:,:,:]  #shape (128,128,128)
    return predmask_arr
    
    
        
def func_mask_back2iso(predmask_arr, cropinfo_df):
    res_orig = cropinfo_df.loc[:, 'res_orig'].item()
    x_min2 = cropinfo_df.loc[:, 'x_min2'].item()
    x_max2 = cropinfo_df.loc[:, 'x_max2'].item()
    x_len = cropinfo_df.loc[:, 'x_len'].item()
    
    y_min2 = cropinfo_df.loc[:, 'y_min2'].item()
    y_max2 = cropinfo_df.loc[:, 'y_max2'].item()
    y_len = cropinfo_df.loc[:, 'y_len'].item()

    z_min2 = cropinfo_df.loc[:, 'z_min2'].item()
    z_max2 = cropinfo_df.loc[:, 'z_max2'].item()
    z_len = cropinfo_df.loc[:, 'z_len'].item()
    
    predmask_arr2 = resampleit(predmask_arr, res_orig, isseg=True)
    predmask_orires_arr = np.zeros((x_len, y_len, z_len))
    predmask_orires_arr[x_min2:x_max2,
                        y_min2:y_max2,
                        z_min2:z_max2] = predmask_arr2
    
    return predmask_orires_arr
    


def func_norm_resnet(img_arr, roi_mask_arr, brain_mask_arr, cropdown_info):
    
    nlbrain_mask_arr = brain_mask_arr - roi_mask_arr
    img_nlbrain_arr = img_arr * nlbrain_mask_arr  
    mean = np.mean(img_nlbrain_arr[img_nlbrain_arr > 0])
    sd = np.std(img_nlbrain_arr[img_nlbrain_arr > 0])

    img_norm_arr = (img_arr - mean) / sd
    img_norm_arr[img_norm_arr < - 5] = - 5
    img_norm_arr[img_norm_arr >   5] = 5
    img_norm_arr = img_norm_arr / 10
    img_norm_arr = img_norm_arr + 0.5
    img_norm_arr = img_norm_arr * brain_mask_arr  ## assign 0 to the background
     
    img_norm_cropdown_arr = func_img_cropdown(img_norm_arr, cropdown_info)
       
    
    return img_norm_cropdown_arr
        
    
    
    
def get_maxROI(t1c_resnet_arr, t2_resnet_arr, mask_arr):
    
    t1c_arr = np.expand_dims(t1c_resnet_arr, axis = 0)  # (1,128,128,128)
    
    t2_arr = np.expand_dims(t2_resnet_arr, axis=0)  # (1,128,128,128)
    
    img_arr = np.concatenate([t1c_arr, t2_arr], axis=0) #shape(2,128,128,128)
    
    truth_arr  = np.expand_dims(mask_arr, axis=0) # shape(1,128,128,128)

     ## find the largest ROI slice
    arr = np.empty((0,2), int)  
    for z in range(truth_arr.shape[2]):  
        slice_sum = np.sum(truth_arr[:,:,:,z])
        arr = np.append(arr, np.array([[z, slice_sum]]), axis=0)    
    z_maxroi = np.argmax(arr[:,1])

    # get images +-2 and 4 slices from maximum tumor slice
    arr_nonzero = arr[arr[:,1]>0]
    z_lowlim = arr_nonzero[0,0]
    z_uplim = arr_nonzero[-1,0]
    
    z_low1 = z_maxroi-4
    z_up1 = z_maxroi+4

    z_low2 = z_maxroi-2
    z_up2 = z_maxroi+2
    
    x_arr_rois = []
    for z2 in [z_low1, z_low2, z_maxroi, z_up1, z_up2]:
        img_arr_maxroi = img_arr[:,:,:,z2]
        truth_arr_maxroi = truth_arr[:,:,:,z2]
        img_arr_maxroi = np.concatenate((img_arr_maxroi, truth_arr_maxroi), axis = 0)   #shape (3,128,128,128)
        img_arr_maxroi = np.expand_dims(img_arr_maxroi, axis=0)  #shape (1, 3,128,128,128)

        x_arr_rois.append(img_arr_maxroi) 

                
    x_arr_rois = np.vstack(x_arr_rois)   #shape (5, 3,128,128,128)
        
    return x_arr_rois


    


################ shape feature extraction #################
import pandas as pd
from radiomics import firstorder, glcm, glrlm, glszm, ngtdm, gldm, shape

def func_shape(mask_arr):
    kwargs_3d = {'binWidth': 1, 'interpolator': None, 'resampledPixelSpacing': None, 
                 'verbose': False, 'force2D':False}

    
    image = sitk.GetImageFromArray(mask_arr)
    mask = sitk.GetImageFromArray(mask_arr)

    shapeFeatures3d = shape.RadiomicsShape(image, mask, **kwargs_3d)
    shapeFeatures3d.enableAllFeatures()
    shapeFeatures3d.execute()
    df_shape = pd.DataFrame(shapeFeatures3d.featureValues, index=[0])
    df_shape = df_shape.add_prefix('Shape_3d_')
    
    return df_shape



def func_regi2mni(path_T1C_isovoxel, path_mask_isovoxel):
    
    matrix_2mni = 'matrix_2mni.mat'
    mni_reference = 'MNI152_T1_1mm_brain.nii.gz'
    
    coregi_t1gd2mni= FLIRT(bins=640, cost_func='mutualinfo', dof=12, output_type="NIFTI_GZ", verbose=0,
                           datatype = 'float', interp = 'trilinear',
                           in_file = path_T1C_isovoxel, 
                           reference = mni_reference,
                           out_file = 'img_2mni.nii.gz',
                           out_matrix_file = matrix_2mni)
    coregi_t1gd2mni.run()
    
    coregi_mask2MNI = ApplyXFM(in_file = path_mask_isovoxel,
                                in_matrix_file = matrix_2mni,
                                out_file = 'mask_2mni.nii.gz',
                                reference= mni_reference)
   
    coregi_mask2MNI.run()
    
import subprocess

def func_loci(path_T1C_isovoxel, path_mask_isovoxel):
    
    #print("Registering tumor mask to MNI space...")
    #func_regi2mni(path_T1C_isovoxel, path_mask_isovoxel)
    
    loci_frame = pd.DataFrame(columns=["mni_Caudate", "mni_Cerebellum", "mni_Frontal Lobe",
                                       "mni_Insula",  "mni_Occipital Lobe", "mni_Parietal Lobe",
                                       "mni_Putamen", "mni_Temporal Lobe",  "mni_Thalamus"])
    print("Acquiring tumor loci information...")
    
    mask_2mni_file = os.path.join(os.getcwd(), 'mask_2mni.nii.gz')
    print(os.path.isfile(mask_2mni_file))
    print(mask_2mni_file)
    command = 'atlasquery -a "MNI Structural Atlas" -m %s'%mask_2mni_file
    print(command)
    get_loci =  subprocess.Popen(command, stdout=subprocess.PIPE, stderr=None, shell=True)
    loci = get_loci.communicate()[0]  ## get atlasquery output from command
    loci= loci.decode()
    print(loci)
    loci_dic = {}
    
    length = loci.count('\n')
    print(length)
    for j in range(length):
        loci_sub = loci.splitlines()[j]
        ind = loci_sub.find(":") 
        print(loci_sub)
        print(ind)
        loci_k = "mni_%s"%(loci_sub[:ind])
        loci_v = float(loci_sub[ind+1 :])
        loci_dic[loci_k] = loci_v
    
    print(loci_dic)
    loci_table = pd.DataFrame([loci_dic])
    loci_table = pd.concat([loci_frame, loci_table], axis=0, sort=True)
    loci_table.fillna(0, inplace=True)
    return loci_table


def func_shapeloci(path_T1C_isovoxel, path_mask_isovoxel):
    mask = nb.load(path_mask_isovoxel)
    mask_arr = mask.get_data()
    df_shape = func_shape(mask_arr)
    df_loci = func_loci(path_T1C_isovoxel, path_mask_isovoxel)
    df_shapeloci = pd.concat([df_shape, df_loci], axis =1)
    
    cols =  ['Shape_3d_Elongation', 'Shape_3d_Flatness',
       'Shape_3d_LeastAxisLength', 'Shape_3d_MajorAxisLength',
       'Shape_3d_Maximum2DDiameterColumn', 'Shape_3d_Maximum2DDiameterRow',
       'Shape_3d_Maximum2DDiameterSlice', 'Shape_3d_Maximum3DDiameter',
       'Shape_3d_MeshVolume', 'Shape_3d_MinorAxisLength',
       'Shape_3d_Sphericity', 'Shape_3d_SurfaceArea', 'Shape_3d_VoxelVolume',
       'mni_Frontal Lobe', 'mni_Insula', 'mni_Occipital Lobe',
       'mni_Parietal Lobe', 'mni_Temporal Lobe', 'mni_Thalamus']
    df_shapeloci = df_shapeloci[cols]
    return df_shapeloci
    
    
##############

    
    


from resnet_model import *    
def get_IDH_pred(t1c_resnet_arr, t2_resnet_arr, mask_arr, sla_arr):
    
    print("Calling pretrained Model 2...")
    model_pre = ResNet(3, BasicBlock, [3,4,6,3])  
    model = ResNet_transfer(model_pre, 2, 20)
    model_filename = 'MODEL2_CNNclassifier.pth'
    checkpoint = torch.load(model_filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.cuda()
    
    print("Calculating predicted probabilitiy...")
    x_arr = get_maxROI(t1c_resnet_arr, t2_resnet_arr, mask_arr)
    x_arr = torch.from_numpy(x_arr).float()
    x_arr = x_arr.cuda()
    
    sla_arr = torch.from_numpy(sla_arr).float()
    sla_arr = sla_arr.cuda()
    
    with torch.no_grad():
        output = model(x_arr, sla_arr)
    
    output = nn.Softmax(dim=1)(output)
    
    output_mean = torch.sum(output[:,1])/5
    print("IDH mutation probability: %s" %(output_mean.cpu().item()*100)
    
    return output_mean
    
