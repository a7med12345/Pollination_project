import torch
import transforms as T
from pollination_model import get_model_instance_segmentation
from PIL import Image


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)

def main(path_to_image):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes=2
    model = get_model_instance_segmentation(num_classes)
    # move model to the right device
    model.to(device)

    model.load_state_dict(torch.load('weights.pth'))
    model.eval()

    im = Image.open(path_to_image)
    transform = get_transform(train=False)

    img,_ = transform(im,im)

    with torch.no_grad():
        prediction = model([img.to(device)])
    Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy()).show()
    masks = prediction[0]['masks']
    n,_,_,_=masks.shape
    x=[]
    for i in range(0,n):
        x.append(masks[i,0].mul(255).byte().cpu().numpy())

    Image.fromarray(sum(x)).show()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Get image path')
    parser.add_argument('--p',  type=str,default='input.png',help='path to image')

    args = parser.parse_args()

    main(args.p)