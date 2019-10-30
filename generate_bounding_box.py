import os
import cv2
from tqdm import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.measure import label, regionprops
#from skimage.util.montage import montage2d as montage
#montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)
#ship_dir = '../input'
#train_image_dir = os.path.join(ship_dir, 'train_v2')
#test_image_dir = os.path.join(ship_dir, 'test_v2')
from PIL import Image
from skimage.morphology import label

def multi_rle_encode(img):
    labels = label(img[:, :, 0])
    return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def masks_as_image(in_mask_list, all_masks=None):
    # Take the individual ship masks and create a single mask array for all ships
    if all_masks is None:
        all_masks = np.zeros((768, 768), dtype = np.int16)
    #if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)


#masks = pd.read_csv(os.path.join('../input/',
   #                              'train_ship_segmentations_v2.csv'))
#print(masks.shape[0], 'masks found')
#print(masks['ImageId'].value_counts().shape[0])
#masks.head()

#images_with_ship = masks.ImageId[masks.EncodedPixels.isnull()==False]
#images_with_ship = np.unique(images_with_ship.values)
#print('There are ' +str(len(images_with_ship)) + ' image files with masks')

#def gnerate_different_masks(mask_path,bbox):
    #for b in bbox:

def generate_bbox_mask(mask_path):
    bbox=[]
    masks =[]
    #img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path)
    mask = np.array(mask)
    #img = np.array(img)

    lbl_0 = label(mask[..., 0])
    props = regionprops(lbl_0)
    #img_1 = img_0.copy()

    for prop in props:
        #area = (prop.bbox[3] - prop.bbox[1]) * (prop.bbox[2] - prop.bbox[0])
        # print(area)
        mask_unique =  mask.copy()*0
        if (prop.bbox_area > 50):
            #print('Found bbox', prop.bbox)
            #cv2.rectangle(img_1, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (255, 0, 0), 2)
            #append bounding ymin,xmin.ymax,xmax
            mask_unique[tuple(list(map(list, zip(*prop.coords))))] = (255, 255, 255)

            bbox.append([prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2]])
            masks.append(mask_unique)

    return bbox,masks

#for i in range(10):
    #image = images_with_ship[i]

    #fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 5))
    #img_0 = cv2.imread('/media/user/751ef869-9928-4453-b951-52ba8a966d63/pollination/Pollination/short_10th/img/short-2019-05-15-15-52-56_70.png')
    #rle_0 = masks.query('ImageId=="'+image+'"')['EncodedPixels']
    #mask_0 = masks_as_image(rle_0)
    #mask_0 = cv2.imread('/media/user/751ef869-9928-4453-b951-52ba8a966d63/pollination/Pollination/short_10th/masks_machine/short-2019-05-15-15-52-56_70.png')
    #print(mask_0[...,0].shape)
    #
    #
    #lbl_0 = label(mask_0[...,0])
    #props = regionprops(lbl_0)
    #img_1 = img_0.copy()
    #print ('Image', image)
    #k=0
    #b=True
    #for prop in props:
        #area = (prop.bbox[3]-prop.bbox[1])*(prop.bbox[2]-prop.bbox[0])
        #print(area)
        #if(area>50):
            #print('Found bbox', prop.bbox)
            #print(area)
            #if(b):
                #for x,y in prop.coords:
                #img_1[tuple(list(map(list, zip(*prop.coords))))] = (255, 255, 255)
                #b=False
            #cv2.rectangle(img_1, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (255, 0, 0), 2)
        #k+=1


    #ax1.imshow(img_0)
    #ax1.set_title('Image')
    #ax2.set_title('Mask')
    #ax3.set_title('Image with derived bounding box')
    #ax2.imshow(mask_0[...,0], cmap='gray')
    #ax3.imshow(img_1)
    #plt.show()



#bboxs,masks = generate_bbox_mask('/media/user/751ef869-9928-4453-b951-52ba8a966d63/pollination/Pollination/short_10th/masks_machine/short-2019-05-15-15-52-56_70.png')
#cv2.rectangle(masks[0], (bboxs[0][0], bboxs[0][1]), (bboxs[0][2], bboxs[0][3]), (255, 0, 0), 2)
#plt.imshow(masks[0])
#plt.show()
import shutil
def copy(root='/media/user/751ef869-9928-4453-b951-52ba8a966d63/pollination/Pollination/short_10th/'):
    imgs = list(sorted(os.listdir(os.path.join(root, "img"))))
    masks = list(sorted(os.listdir(os.path.join(root, "masks_machine"))))
    for i,m in zip(imgs,masks):
        img_path = os.path.join(root, "img", i)
        mask_path = os.path.join(root, "masks_machine",m)
        bboxs,_ = generate_bbox_mask(mask_path)
        dst_im=os.path.join('/media/user/751ef869-9928-4453-b951-52ba8a966d63/pollination/Pollination/short_10th/data/', "img1", i)
        dst_mask=os.path.join('/media/user/751ef869-9928-4453-b951-52ba8a966d63/pollination/Pollination/short_10th/data/', "mask1", i)
        if(len(bboxs)>0):
            shutil.copy(img_path, dst_im)
            shutil.copy(mask_path, dst_mask)


copy()
