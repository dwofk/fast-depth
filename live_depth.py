import torch
import torchvision
import argparse
import utils
import models
import cv2
from torch.utils import data
from torchvision import transforms
import numpy as np

# global display

def draw_circle(event,x,y,flags,param):
    global unscaled_depth
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print("Depth: {} m".format(unscaled_depth[y][x]))

parser = argparse.ArgumentParser(description='FastDepth evaluation')
parser.add_argument('-m', '--model', type=str, required=True, help="Path to model.")
args = parser.parse_args()

model_path = args.model

model_state_dict, _, _, _ = utils.load_checkpoint(args.model)
model_state_dict = utils.convert_state_dict_from_gpu(model_state_dict)
model = models.MobileNetSkipAdd(output_size=(224, 224), pretrained=True)
if model_state_dict:
    model.load_state_dict(model_state_dict)
device = torch.device("cuda:0")
model.to(device)

cap = cv2.VideoCapture(0)
to_tensor = transforms.ToTensor()

out = cv2.VideoWriter("live_depth.avi", cv2.VideoWriter_fourcc(*'MJPG'), 25.0, (224*2, 224))

display = np.array([])
cv2.namedWindow('Live Depth')
cv2.setMouseCallback('Live Depth',draw_circle, display)

while True:
    _, frame = cap.read()
    frame = cv2.normalize(frame, frame, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    frame = cv2.resize(frame, (224, 224))
    permutated_channels = np.random.permutation(3)
    pframe = frame[:, :, permutated_channels]
    
    input = to_tensor(frame).to(device).unsqueeze(0)
    pinput = to_tensor(pframe).to(device).unsqueeze(0)

    depth_pred = model(input)
    pdepth_pred = model(pinput)

    nframe = np.array([])
    nframe = cv2.normalize(pframe, nframe, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    unscaled_depth = np.squeeze(depth_pred.data.cpu().numpy())
    unscaled_depth[unscaled_depth > 25.0] = 25.0
    
    depth = unscaled_depth.copy()
    depth = cv2.normalize(depth, depth, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
    depth = np.array(depth * 255, dtype=np.uint8)
    # depth = cv2.applyColorMap(depth, cv2.COLORMAP_TURBO)

    # pdepth = np.squeeze(pdepth_pred.data.cpu().numpy())
    # pdepth = cv2.normalize(pdepth, pdepth, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    # pdepth = cv2.cvtColor(pdepth, cv2.COLOR_GRAY2BGR)
    # pdepth = np.array(pdepth * 255, dtype=np.uint8)
    # depth = cv2.applyColorMap(depth, cv2.COLORMAP_TURBO)
    
    display = np.hstack([nframe, depth])
    
    cv2.imshow("Live Depth", depth)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    # out.write(display)
