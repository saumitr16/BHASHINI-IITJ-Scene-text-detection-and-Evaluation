import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFile
from model import CRAFT
from config import cfg

# Allow truncated images to be loaded
ImageFile.LOAD_TRUNCATED_IMAGES = True

def plot_boxes(img, boxes):
    if boxes is None:
        return img
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.polygon([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]], outline=(0, 255, 0))
    return img

def resize_img(img, long_side):
    w, h = img.size
    if long_side is not None:
        if w > h:
            resize_w = long_side
            ratio = long_side / w
            resize_h = h * ratio
        else:
            resize_h = long_side
            ratio = long_side / h
            resize_w = w * ratio
    else:
        resize_h, resize_w = h, w

    final_h = int(resize_h) if resize_h % 32 == 0 else (int(resize_h / 32) + 1) * 32
    final_w = int(resize_w) if resize_w % 32 == 0 else (int(resize_w / 32) + 1) * 32
    img = img.resize((final_w, final_h), Image.BILINEAR)
    ratio_h = final_h / h
    ratio_w = final_w / w
    return img, ratio_h, ratio_w

def load_pil(img):
    # Convert grayscale images to RGB by repeating the single channel three times
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cfg.train.mean, cfg.train.std)])
    return t(img).unsqueeze(0)

def get_score(img, model, device):
    with torch.no_grad():
        region, affinity = model(load_pil(img).to(device))
    return list(map(lambda x: x[0][0].cpu().numpy(), [region, affinity]))

def restore_boxes(region, affinity, region_thresh, affinity_thresh, remove_thresh, ratio):
    boxes = []
    M = (region > region_thresh) + (affinity > affinity_thresh)
    ret, markers = cv2.connectedComponents(np.uint8(M * 255))
    for i in range(ret):
        if i == 0:
            continue
        y, x = np.where(markers == i)
        if len(y) < region.size * remove_thresh:
            continue
        cords = 2 * np.concatenate((x.reshape(-1, 1) / ratio[1], y.reshape(-1, 1) / ratio[0]), axis=1)
        a = np.array([cords[:, 0].min(), cords[:, 1].min(), cords[:, 0].max(), cords[:, 1].min(), cords[:, 0].max(), cords[:, 1].max(), cords[:, 0].min(), cords[:, 1].max()])
        boxes.append(a)
    return boxes

def detect_single_image(img, model, device, cfg):
    img, ratio_h, ratio_w = resize_img(img, cfg.long_side)
    region, affinity = get_score(img, model, device)
    boxes = restore_boxes(region, affinity, cfg.region_thresh, cfg.affinity_thresh, cfg.remove_thresh, (ratio_h, ratio_w))
    return boxes

def save_boxes(boxes, file_path):
    with open(file_path, 'w') as f:
        for box in boxes:
            x_min = min(box[0], box[2], box[4], box[6])
            x_max = max(box[0], box[2], box[4], box[6])
            y_min = min(box[1], box[3], box[5], box[7])
            y_max = max(box[1], box[3], box[5], box[7])
            f.write(f'{int(x_min)}, {int(y_min)}, {int(x_max)}, {int(y_max)}\n')

if __name__ == '__main__':
    images_folder = 'urdu\est_images'
    result_folder = 'urdu\est_result'
    os.makedirs(result_folder, exist_ok=True)

    model_path = './pths/pretrain/model_iter_50000.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CRAFT().to(device)

    map_location = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load(model_path, map_location=map_location))

    model.eval()

    for img_name in os.listdir(images_folder):
        img_path = os.path.join(images_folder, img_name)
        try:
            img = Image.open(img_path)
            boxes = detect_single_image(img, model, device, cfg.test)

            result_path = os.path.join(result_folder, f'res_{os.path.splitext(img_name)[0]}.txt')
            save_boxes(boxes, result_path)
            
            print(f'Processed {img_name} and saved results to {result_path}')
        except Exception as e:
            print(f'Error processing {img_name}: {e}')
