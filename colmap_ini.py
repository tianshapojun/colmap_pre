# 该脚本是为了通过已知内外参构建colmap配置文件，从而根据已知位姿输入colmap进行特征点匹配

import numpy as np
import json
import os
#import imageio
import math
import cv2

from utils.camera_utils import Calibration
from utils.imu_utils import load_pointsclouds_new,Calibration_W2C,Calibration2_W2C

folder = 'test_01'
root_dir = os.path.dirname(os.path.abspath(__file__))
idx_list = [i for i in range(331,391)]

calib_filename = os.path.join(root_dir,'Data',folder,'calib')
calib = Calibration(calib_filename, from_origin=True)

img_filename = os.path.join(root_dir,'Data',folder,'image_2', '{}.png'.format("%010d" % (idx_list[0])))
img = cv2.imread(img_filename)
H, W, _ = img.shape
poses = load_pointsclouds_new(root_dir, calib, folder, idx_list, poses_only = True)

with open(os.path.join(root_dir,'Data',folder,'sparse','cameras.txt'), 'w') as f:
    f.write(f'1 PINHOLE {W} {H} {calib.P_new[0,0]} {calib.P_new[1,1]} {calib.P_new[0,2]} {calib.P_new[1,2]}')

with open(os.path.join(root_dir,'Data',folder,'sparse','images.txt'), 'w') as f:
    i = 0
    for fname in idx_list:
        pose = poses[i]
        calib_c = Calibration_W2C(calib, pose[:3,:4], H, W)
        R = calib_c.W2C[:3,:3]
        T = calib_c.W2C[:3,3]
        q0 = 0.5 * math.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
        q1 = (R[2, 1] - R[1, 2]) / (4 * q0)
        q2 = (R[0, 2] - R[2, 0]) / (4 * q0)
        q3 = (R[1, 0] - R[0, 1]) / (4 * q0)

        f.write(f'{2*i+1} {q0} {q1} {q2} {q3} {T[0]} {T[1]} {T[2]} {1} {"%06d.png" % (fname)}\n\n')
        
        calib_c = Calibration2_W2C(calib, pose[:3,:4], H, W)
        R = calib_c.W2C[:3,:3]
        T = calib_c.W2C[:3,3]
        q0 = 0.5 * math.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
        q1 = (R[2, 1] - R[1, 2]) / (4 * q0)
        q2 = (R[0, 2] - R[2, 0]) / (4 * q0)
        q3 = (R[1, 0] - R[0, 1]) / (4 * q0)

        f.write(f'{2*i+2} {q0} {q1} {q2} {q3} {T[0]} {T[1]} {T[2]} {1} {"%010d.png" % (fname)}\n\n')
        
        i += 1

with open(os.path.join(root_dir,'Data',folder,'sparse','points3D.txt'), 'w') as f:
   f.write('')
