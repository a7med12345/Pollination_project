import os
import numpy as np
import torch
from PIL import Image
from skimage.measure import label, regionprops


class PollinationDataset(object):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "img"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "mask"))))

    def get_bbox_mask(self,mask_path):
        bbox = []
        masks = []
        areas = []
        # img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)
        # img = np.array(img)

        lbl_0 = label(mask[..., 0])
        props = regionprops(lbl_0)
        # img_1 = img_0.copy()

        for prop in props:
            # area = (prop.bbox[3] - prop.bbox[1]) * (prop.bbox[2] - prop.bbox[0])
            # print(area)
            mask_unique = mask[..., 0].copy() * 0
            if (prop.bbox_area > 50):
                # print('Found bbox', prop.bbox)
                # cv2.rectangle(img_1, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (255, 0, 0), 2)
                # append bounding ymin,xmin.ymax,xmax
                mask_unique[tuple(list(map(list, zip(*prop.coords))))] = 1
                #mask_unique = np.transpose(mask_unique)
                xmin_new = min(prop.bbox[0],prop.bbox[2])
                xmax_new=  max(prop.bbox[0],prop.bbox[2])

                ymin_new = min(prop.bbox[1], prop.bbox[3])
                ymax_new = max(prop.bbox[1], prop.bbox[3])

                bbox.append([prop.bbox[0], prop.bbox[1], prop.bbox[2], prop.bbox[3]])
                #bbox.append([xmin_new, ymin_new, xmax_new, ymax_new ])
                masks.append(mask_unique)
                areas.append(prop.bbox_area)

        return bbox, masks,areas

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "img", self.imgs[idx])
        mask_path = os.path.join(self.root, "mask", self.masks[idx])
        img = Image.open(img_path).convert("RGB")



        boxes, masks,area = self.get_bbox_mask(mask_path)
        num_objs = len(boxes)
        #for b in boxes:
            #print('xmin',b[0])
            #print('xmax',b[2])
            #print('ymin', b[1])
            #print('ymax', b[3])
            #print("------------------")
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        area = torch.as_tensor(area, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        #area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        #return 10
        return len(self.imgs)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    #import cv2

    dataset = PollinationDataset('/media/user/751ef869-9928-4453-b951-52ba8a966d63/pollination/Pollination/short_10th/')
    im = dataset[0][0]
    m  =  dataset[0][1]["masks"][0].numpy()
    b  =  dataset[0][1]["boxes"][0].numpy()
    area = dataset[0][1]["area"].numpy()
    #cv2.rectangle(m, (b[3], b[1]), (b[2], b[0]), (255, 0, 0), 2)
    #print(area)
    print(m.shape)
    plt.imshow(m)
    plt.show()