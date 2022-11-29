# 2022 고급딥러닝시스템 응용 실습
1. NICE-SLAM + Point-NeRF : Point Feature를 MLP 입력으로 추가
2. NICE-SLAM + Monocular Depth Estimation : RGB-D 데이터셋을 Mono Depth Estimation으로 교체

# 2022-11-22 Class
## Colab 사용해서 nice-slam 실행하기
1. git clone

(optional). google drive mount 
```
from google.colab import drive
drive.mount('/content/drive')
```
(optional). google colab run time 유지: Press F12
```
function ClickConnect(){
console.log("Working"); 
document.querySelector("colab-toolbar-button#connect").click() 
}setInterval(ClickConnect, 1800000)
```


2. prepare environments
```
!apt-get install libopenexr-dev
!pip install colorama open3d trimesh rtree mathutils==2.81.2
```

3. demo download
```
cd nice-slam
!bash scripts/download_demo.sh
```

4. demo 돌려보기
```
!python -W ignore run.py configs/Demo/demo.yaml
```

## 코드 읽고 직접 손으로 모델 그려보기
1. nice-slam 코드에서 dataset 불러오기
```python
import argparse
from src.utils.datasets import ScanNet
from src.config import load_config
import matplotlib.pyplot as plt

args = argparse.Namespace(config = 'configs/Demo/demo.yaml', \
                          input_folder = None, output = None, nice = True)
cfg = load_config(args.config, 'configs/nice_slam.yaml')

frame_reader = ScanNet(cfg, args, cfg['scale'], device ='cuda:0') # device='cpu')
n_img = len(frame_reader)

for idx, gt_color, gt_depth, gt_c2w in frame_reader:
    break

plt.figure(figsize=[11,5])
plt.subplot(121); plt.imshow(gt_color.cpu());
plt.subplot(122); plt.imshow(gt_depth.cpu()); 

idx, gt_color.shape, gt_depth.shape, gt_c2w.shape
```
2. Get Models

```
@ src\conv_onet\config.py
```

<!--
<details>
```python
from src.conv_onet.models.decoder import NICE

dim = cfg['data']['dim']
coarse_grid_len = cfg['grid_len']['coarse']
middle_grid_len = cfg['grid_len']['middle']
fine_grid_len = cfg['grid_len']['fine']
color_grid_len = cfg['grid_len']['color']
c_dim = cfg['model']['c_dim']  # feature dimensions
pos_embedding_method = cfg['model']['pos_embedding_method']
decoder = NICE(
    dim=dim, c_dim=c_dim, coarse=cfg['coarse'], coarse_grid_len=coarse_grid_len,
    middle_grid_len=middle_grid_len, fine_grid_len=fine_grid_len,
    color_grid_len=color_grid_len, pos_embedding_method=pos_embedding_method)
decoder
```  
</details>
-->

4. Sampling Ray 가져오기, Get Ray Samples

```
@ src\Tracker.py
```

```python
from src.NICE_SLAM import NICE_SLAM

self = NICE_SLAM(cfg, args)

batch_size = tracking_pixels = cfg['tracking']['pixels']

# ...

import torch
device = self.tracker.device

```

5. Renderer 돌려보기, Run Renderder

```
@ src\Renderer.py
```
```python
self.tracker.c = {}
self.tracker.prev_mapping_idx += 1
self.tracker.update_para_from_mapping()

ret = self.renderer.render_batch_ray(
    self.tracker.c, self.tracker.decoders, batch_rays_d, batch_rays_o, \
     self.tracker.device, stage='color',  gt_depth=batch_gt_depth)
depth, uncertainty, color = ret
```

```python
# def render_batch_ray(self, c, decoders, rays_d, rays_o, device, stage, gt_depth=None):

c, decoders, rays_d, rays_o, device, stage, gt_depth = \
self.tracker.c, self.tracker.decoders, batch_rays_d, batch_rays_o, \
 self.tracker.device, 'color',  batch_gt_depth


N_samples = self.renderer.N_samples
N_surface = self.renderer.N_surface
N_importance = self.renderer.N_importance

# ...
```
6. 모델 summary 확인하기
```python
# def eval_points(self, p, decoders, c=None, stage='color', device='cuda:0'):
p, decoders, c, stage, device = pointsf, decoders, c, stage, device

p_split = torch.split(p, self.renderer.points_batch_size) # points_batch_size=500000 @ class Renderer(object):  def __init__(..
bound = self.bound
rets = []
for pi in p_split:
    break
```
```
!pip install torchinfo
```

```python
decoder.middle_decoder.bound = self.bound
decoder.color_decoder.bound = self.bound
decoder.fine_decoder.bound = self.bound

ret = decoder(pi, c_grid=c, stage=stage) # class NICE(nn.Module): def forward(self, p, c_grid, stage='middle', **kwargs):
```

```python
from torchinfo import summary
summary(decoder, input_data = [pi, c])
```

## Decoder model 직접 구현하기
1. Decoder model 구조 손으로 그려보기
- 구조도 그림 예시:
<!--
![image](https://user-images.githubusercontent.com/10238769/203263221-b7cfa34d-2548-408a-a671-52c473e63a72.png){: width="100" height="100"}
-->
<img src="https://user-images.githubusercontent.com/10238769/203263221-b7cfa34d-2548-408a-a671-52c473e63a72.png" width="200" height="200"/>
                                                                                                                                         
2. 직접 구현하기

## 프로젝트 과제 1 (12/06) 수업시간 까지
> #### 과제 내용 
> - class MLP(nn.Module): 안에 있는 __init__ 함수의 self.fc_c와 self.pts_linears를 for문 없이 구현
> - DenseLayer 함수없이 nn.Layer로만 구현하고 DenseLayer 안에 parameter 초기화 부분은 무시함
> - class MLP(nn.Module): 안에 있는 __forward__ 함수의 for i, l in enumerate(self.pts_linears) 부분 for문 없이 구현
> #### 결과 확인 방법 
> - points와 color grid를 넣었을 때 예:
> - ```python ret = decoder(pi, c_grid=c, stage=stage)``` 
> - ret.shape가 torch.Size([48000, 4]) 임을 확인
> - 코드 및 결과 정리한 PPT 파일 제출 (수업시간에 코드 확인)

# 2022-11-29 Class
## 지난시간 복습
1. pytorch3d & nice-slam install
```python
# pytorch3d install
import torchvision
!pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
!pip install fvcore iopath 
import sys
import torch
pyt_version_str=torch.__version__.split("+")[0].replace(".", "")
version_str="".join([
    f"py3{sys.version_info.minor}_cu",
    torch.version.cuda.replace(".",""),
    f"_pyt{pyt_version_str}"
])
!pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html

# nice-slam requirements
!git clone https://github.com/jooyongsim/nice-slam.git
!apt-get install libopenexr-dev
!pip install colorama open3d trimesh rtree mathutils==2.81.2
%cd nice-slam
!bash scripts/download_demo.sh
```

2. Load Dataset
```python
import argparse
from src.utils.datasets import ScanNet
from src.config import load_config
import matplotlib.pyplot as plt

args = argparse.Namespace(config = 'configs/Demo/demo.yaml', \
                          input_folder = None, output = None, nice = True)
cfg = load_config(args.config, 'configs/nice_slam.yaml')

frame_reader = ScanNet(cfg, args, cfg['scale'], device ='cuda:0') # device='cpu')
n_img = len(frame_reader)

for idx, gt_color, gt_depth, gt_c2w in frame_reader:
    break

plt.figure(figsize=[11,5])
plt.subplot(121); plt.imshow(gt_color.cpu());
plt.subplot(122); plt.imshow(gt_depth.cpu()); 

depth_data, color_data = gt_depth, gt_color
```

3. Load Models & NICE-SLAM Class
```python
# @ src/conv_onet/config.py
%load_ext autoreload
%autoreload 2

from src.conv_onet.models.decoder import NICE 

dim = cfg['data']['dim']
coarse_grid_len = cfg['grid_len']['coarse']
middle_grid_len = cfg['grid_len']['middle']
fine_grid_len = cfg['grid_len']['fine']
color_grid_len = cfg['grid_len']['color']
c_dim = cfg['model']['c_dim']  # feature dimensions
pos_embedding_method = cfg['model']['pos_embedding_method']

decoder = NICE(
    dim=dim, c_dim=c_dim, coarse=cfg['coarse'], coarse_grid_len=coarse_grid_len,
    middle_grid_len=middle_grid_len, fine_grid_len=fine_grid_len,
    color_grid_len=color_grid_len, pos_embedding_method=pos_embedding_method)

from src.NICE_SLAM import NICE_SLAM
self = NICE_SLAM(cfg, args)
```

4. Get Sampling Rays 
```python
# @ src/Tracker.py
from src.common import (get_camera_from_tensor, get_samples,
                        get_tensor_from_camera)

batch_size = cfg['tracking']['pixels'] # 1000
c2w = gt_c2w
device = self.tracker.device
H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
# optimizer.zero_grad()
# c2w = get_camera_from_tensor(camera_tensor)
Wedge = self.tracker.ignore_edge_W
Hedge = self.tracker.ignore_edge_H
batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(
    Hedge, H-Hedge, Wedge, W-Wedge, batch_size, H, W, fx, fy, cx, cy, c2w, \
    gt_depth, gt_color, self.tracker.device)

import torch
with torch.no_grad():
    det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
    det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
    t = (self.bound.unsqueeze(0).to(device)-det_rays_o)/det_rays_d
    t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
    inside_mask = t >= batch_gt_depth
batch_rays_d = batch_rays_d[inside_mask]
batch_rays_o = batch_rays_o[inside_mask]
batch_gt_depth = batch_gt_depth[inside_mask]
batch_gt_color = batch_gt_color[inside_mask]

def print_shape(*paras):
    for para in paras:
        print(para.shape)

print_shape(batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color) 
Hedge, H-Hedge, Wedge, W-Wedge, self.H, self.W, self.fx, self.fy, self.cx, self.cy
```

5. Render Batch Ray
```python
# @ src/Renderer.py
self.tracker.c = {}  # self.tracker.prev_mapping_idx += 1
self.tracker.update_para_from_mapping()
print(self.tracker.c.keys())

ret = self.renderer.render_batch_ray(
    self.tracker.c, self.tracker.decoders, batch_rays_d, batch_rays_o, \
     self.tracker.device, stage='color', gt_depth=batch_gt_depth)
depth, uncertainty, color = ret

print_shape(self.tracker.c['grid_coarse'], self.tracker.c['grid_middle'], \
self.tracker.c['grid_fine'], self.tracker.c['grid_color'], )

print_shape(depth, uncertainty, color )
```

6. 직접 sampling ray 만들어보기
```python
# @ src/Renderer.py
# @ src/Renderer.py
c, decoders, rays_d, rays_o, device, stage, gt_depth = \
self.tracker.c, self.tracker.decoders, batch_rays_d, batch_rays_o, \
 self.tracker.device, 'color',  batch_gt_depth

N_samples = self.renderer.N_samples
N_surface = self.renderer.N_surface
N_importance = self.renderer.N_importance

N_rays = rays_o.shape[0]

if stage == 'coarse':
    gt_depth = None
if gt_depth is None:
    N_surface = 0
    near = 0.01
else:
    gt_depth = gt_depth.reshape(-1, 1)
    gt_depth_samples = gt_depth.repeat(1, N_samples)
    near = gt_depth_samples*0.01

with torch.no_grad():
    det_rays_o = rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
    det_rays_d = rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
    t = (self.bound.unsqueeze(0).to(device) -
            det_rays_o)/det_rays_d  # (N, 3, 2)
    far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
    far_bb = far_bb.unsqueeze(-1)
    far_bb += 0.01

if gt_depth is not None:
    # in case the bound is too large
    far = torch.clamp(far_bb, 0,  torch.max(gt_depth*1.2))
else:
    far = far_bb
if N_surface > 0:
    if False:
        # this naive implementation downgrades performance
        gt_depth_surface = gt_depth.repeat(1, N_surface)
        t_vals_surface = torch.linspace(
            0., 1., steps=N_surface).to(device)
        z_vals_surface = 0.95*gt_depth_surface * \
            (1.-t_vals_surface) + 1.05 * \
            gt_depth_surface * (t_vals_surface)
    else:
        # since we want to colorize even on regions with no depth sensor readings,
        # meaning colorize on interpolated geometry region,
        # we sample all pixels (not using depth mask) for color loss.
        # Therefore, for pixels with non-zero depth value, we sample near the surface,
        # since it is not a good idea to sample 16 points near (half even behind) camera,
        # for pixels with zero depth value, we sample uniformly from camera to max_depth.
        gt_none_zero_mask = gt_depth > 0
        gt_none_zero = gt_depth[gt_none_zero_mask]
        gt_none_zero = gt_none_zero.unsqueeze(-1)
        gt_depth_surface = gt_none_zero.repeat(1, N_surface)
        t_vals_surface = torch.linspace(
            0., 1., steps=N_surface).double().to(device)
        # emperical range 0.05*depth
        z_vals_surface_depth_none_zero = 0.95*gt_depth_surface * \
            (1.-t_vals_surface) + 1.05 * \
            gt_depth_surface * (t_vals_surface)
        z_vals_surface = torch.zeros(
            gt_depth.shape[0], N_surface).to(device).double()
        gt_none_zero_mask = gt_none_zero_mask.squeeze(-1)
        z_vals_surface[gt_none_zero_mask,
                        :] = z_vals_surface_depth_none_zero
        near_surface = 0.001
        far_surface = torch.max(gt_depth)
        z_vals_surface_depth_zero = near_surface * \
            (1.-t_vals_surface) + far_surface * (t_vals_surface)
        z_vals_surface_depth_zero.unsqueeze(
            0).repeat((~gt_none_zero_mask).sum(), 1)
        z_vals_surface[~gt_none_zero_mask,
                        :] = z_vals_surface_depth_zero

t_vals = torch.linspace(0., 1., steps=N_samples, device=device)

if not self.renderer.lindisp:
    z_vals = near * (1.-t_vals) + far * (t_vals)
else:
    z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

if self.renderer.perturb > 0.:
    # get intervals between samples
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], -1)
    lower = torch.cat([z_vals[..., :1], mids], -1)
    # stratified samples in those intervals
    t_rand = torch.rand(z_vals.shape).to(device)
    z_vals = lower + (upper - lower) * t_rand

if N_surface > 0:
    z_vals, _ = torch.sort(
        torch.cat([z_vals, z_vals_surface.double()], -1), -1)

pts = rays_o[..., None, :] + rays_d[..., None, :] * \
    z_vals[..., :, None]  # [N_rays, N_samples+N_surface, 3]
pointsf = pts.reshape(-1, 3)
print_shape(t_vals, pointsf, z_vals, z_vals_surface)
```

7. Neural Network에 sampling ray 넣어서 테스트
```python
p, decoders, c, stage, device = pointsf, decoders, c, stage, device

p_split = torch.split(p, self.renderer.points_batch_size) # points_batch_size=500000 @ class Renderer(object):  def __init__(..
bound = self.bound
rets = []
for pi in p_split:
    break

decoder = decoder.to(device)
decoder.middle_decoder.bound = self.bound
decoder.color_decoder.bound = self.bound
decoder.fine_decoder.bound = self.bound

ret = decoder(pi, c_grid=c, stage=stage)
ret.shape
```
## Depth map를 이용해서 point cloud 만들기
1. Load instrinsic
```python
@ src/NICE_SLAM.py
# Load Intrinsic
H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

intrinsic = torch.tensor([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).to()
int_inv = torch.inverse(intrinsic.t().cpu()).to(device)
```
2. Load Depth Map View
```python
# Load Depth Map View
mg_x, mg_y = torch.meshgrid(#..., #..., #...)
mg_x, mg_y = mg_x.to(device), mg_y.to(device)

# Depth Point Cloud 
cam_xyz = torch.stack([mg_x* depth_data, mg_y* depth_data, depth_data], dim=-1)
cam_xyz = # ... #
cam_xyz.shape, color_data.shape
```

3. Depth point cloud 좌표계를 camera 에서 world로 변환
```
# Convert depth point cloud coordinate from camera to world
p_dp = cam_xyz.to(torch.float32).reshape(-1,3)

#...
p_rs = pointsf.to(torch.float32)
```

## Point Cloud 시각화
1. sampling ray point
```python
import open3d as o3d

ray_sampling = o3d.geometry.PointCloud()
ray_sampling.points = o3d.utility.Vector3dVector(p_rs.cpu())
ray_sampling_color = torch.zeros_like(p_rs.cpu().detach()) + torch.Tensor((0,176,0))
ray_sampling.colors = o3d.utility.Vector3dVector(ray_sampling_color)
ray_sampling = ray_sampling.voxel_down_sample(voxel_size=0.2)
```
2. depth point cloud
```python
total_pt = o3d.geometry.PointCloud()
total_pt.points = o3d.utility.Vector3dVector(p_dp.cpu())

total_pt_color = gt_color.cpu().reshape(-1,3)
total_pt.colors = o3d.utility.Vector3dVector(total_pt_color)
total_pt = total_pt.voxel_down_sample(voxel_size=0.05)

o3d.visualization.draw_plotly([total_pt, ray_sampling]) 
```

## kNN Point sampling
1. kNN of depth point cloud for sampling ray
```python
from pytorch3d.ops.knn import knn_points
import time

print_shape(p_rs,p_dp)

s = time.time()
rest = knn_points(p_rs.unsqueeze(0), p_dp.unsqueeze(0), K = 8)

torch.cuda.synchronize()
print(f"Time: {time.time() - s} seconds")
```
2. Visualize one of kNNs
```python
import numpy as np
# kNN of depth point cloud
rand_idx = 9904 
knn_idx = rest.idx[0,rand_idx]
knn_point= p_dp[knn_idx].cpu().detach().numpy()
knn_color = np.tile([0,0,255],(knn_point.shape[0],1))
# ray sampling point
knn_probe = p_rs[rand_idx].cpu().detach().numpy()
knn_probe = knn_probe[None,...]

knn_point = np.concatenate((knn_point, knn_probe),axis = 0)
knn_color = np.concatenate((knn_color, [[255,0,0]]), axis = 0)

print_shape(knn_point, knn_color)
knn_point, knn_color
```
3. Open3D plot
```python
knn_point = torch.Tensor(np.array(knn_point)).reshape(-1,3)
knn_color = torch.Tensor(np.array(knn_color)).reshape(-1,3)

k_querying = o3d.geometry.PointCloud()
k_querying.points = o3d.utility.Vector3dVector(knn_point)
k_querying.colors = o3d.utility.Vector3dVector(knn_color)

o3d.visualization.draw_plotly([total_pt, k_querying])
```

## Pretrained Point Features
### VGG Pretrained Model
```python
import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=True)
vgg16
```

```python
layers = []
for cnt, layer in enumerate(vgg16.features):
    layers.append(layer)
    if cnt == 15:
        break

layers += [# ... ]
vgg_sel = nn.Sequential(*layers)

img = torch.randn((1,3,128,128))
out = vgg_sel(img)

img.shape, out.shape
```
### VGG Input Data Preparation

```python
from torchvision import transforms

data_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
vgg_sel = vgg_sel.to(device)
tf_image = data_transforms(color_data.permute((2,0,1))).to(torch.float32)

vgg_out = vgg_sel(tf_image)
vgg_out.shape, color_data.shape, tf_image.shape
```
### VGG Feature를 (32,32)에서 (H,W)로 Interpolation
- F.grid_sample 이용
```python
H, W = depth_data.shape
gr_x = # ...
gr_y = # ...
# ... torch.meshgrid(...

import torch.nn.functional as F

vgrid = torch.stack([gr_x, gr_y], dim=-1).to(device).unsqueeze(0)

p_feat = F.grid_sample(# ...)
```



<!--
### NDC 좌표계로 변환
```python
valid_x, valid_y = valid_x.to(device), valid_y.to(device)
intrinsic = intrinsic.to(device)

ndc_xyz = torch.stack([valid_x, valid_y, valid_z], dim=-1)

H, W = depth_data.shape
inv_scale = torch.tensor([[W - 1, H - 1]], device=ndc_xyz.device)
cam_z = ndc_xyz[..., 2:3] * (far-near) + near 
cam_xy = ndc_xyz[..., :2] * inv_scale * cam_z 
cam_xyz = torch.cat([cam_xy, cam_z], dim=-1)  
print(cam_xyz.shape, cam_xyz.max(), cam_xyz.min(), cam_xyz.dtype)

cam_xyz = cam_xyz @ torch.inverse(intrinsic.t()).to(torch.float32)
print(cam_xyz.shape, cam_xyz.max(), cam_xyz.min(), cam_xyz.dtype)
```
-->



<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center"><img src="media/logo.png" width="60">NICE-SLAM: Neural Implicit Scalable Encoding for SLAM</h1>
  <p align="center">
    <a href="https://zzh2000.github.io"><strong>Zihan Zhu*</strong></a>
    ·
    <a href="https://pengsongyou.github.io"><strong>Songyou Peng*</strong></a>
    ·
    <a href="http://people.inf.ethz.ch/vlarsson/"><strong>Viktor Larsson</strong></a>
    ·
    <a href="http://www.cad.zju.edu.cn/home/weiweixu/weiweixu_en.htm"><strong>Weiwei Xu</strong></a>
    ·
    <a href="http://www.cad.zju.edu.cn/home/bao/"><strong>Hujun Bao</strong></a>
    <br>
    <a href="https://zhpcui.github.io/"><strong>Zhaopeng Cui</strong></a>
    ·
    <a href="http://people.inf.ethz.ch/moswald/"><strong>Martin R. Oswald</strong></a>
    ·
    <a href="https://people.inf.ethz.ch/pomarc/"><strong>Marc Pollefeys</strong></a>
  </p>
  <p align="center"><strong>(* Equal Contribution)</strong></p>
  <h2 align="center">CVPR 2022</h2>
  <h3 align="center"><a href="https://arxiv.org/abs/2112.12130">Paper</a> | <a href="https://youtu.be/V5hYTz5os0M">Video</a> | <a href="https://pengsongyou.github.io/nice-slam">Project Page</a></h3>
  <div align="center"></div>
</p>
<p align="center">
  <a href="">
    <img src="./media/apartment.gif" alt="Logo" width="80%">
  </a>
</p>
<p align="center">
NICE-SLAM produces accurate dense geometry and camera tracking on large-scale indoor scenes.
</p>
<p align="center">
(The black / red lines are the ground truth / predicted camera trajectory)
</p>
<br>



<br>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#visualizing-nice-slam-results">Visualization</a>
    </li>
    <li>
      <a href="#demo">Demo</a>
    </li>
    <li>
      <a href="#run">Run</a>
    </li>
    <li>
      <a href="#imap">iMAP*</a>
    </li>
    <li>
      <a href="#evaluation">Evaluation</a>
    </li>
    <li>
      <a href="#acknowledgement">Acknowledgement</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
    <li>
      <a href="#contact">Contact</a>
    </li>
  </ol>
</details>


## Installation

First you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

You can create an anaconda environment called `nice-slam`. For linux, you need to install **libopenexr-dev** before creating the environment.
```bash
sudo apt-get install libopenexr-dev
    
conda env create -f environment.yaml
conda activate nice-slam
```

## Visualizing NICE-SLAM Results
We provide the results of NICE-SLAM ready for download. You can run our **interactive visualizer** as following. 

### Self-captured Apartment
To visualize our results on the self-captured apartment, as shown in the teaser:
```bash
bash scripts/download_vis_apartment.sh
python visualizer.py configs/Apartment/apartment.yaml --output output/vis/Apartment
```

**Note for users from China:**  If you encounter slow speed in downloading, check in all the `scripts/download_*.sh` scripts, where we also provide the 和彩云 links for you to download manually.

### ScanNet
```bash
bash scripts/download_vis_scene0000.sh
python visualizer.py configs/ScanNet/scene0000.yaml --output output/vis/scannet/scans/scene0000_00
```
<p align="center">
  <img src="./media/scannet.gif" width="60%" />
</p>

You can find the results of NICE-SLAM on other scenes in ScanNet [here](https://cvg-data.inf.ethz.ch/nice-slam/vis/scannet/).

### Replica
```bash
bash scripts/download_vis_room1.sh
python visualizer.py configs/Replica/room1.yaml --output output/vis/Replica/room1
```
<p align="center">
  <img src="./media/replica.gif" width="70%" />
</p

You can find the results of NICE-SLAM on other scenes in Replica [here](https://cvg-data.inf.ethz.ch/nice-slam/vis/Replica/).

### Interactive Visualizer Usage
The black trajectory indicates the ground truth trajectory, abd the red is trajectory of NICE-SLAM. 
- Press `Ctrl+0` for grey mesh rendering. 
- Press `Ctrl+1` for textured mesh rendering. 
- Press `Ctrl+9` for normal rendering. 
- Press `L` to turn off/on lighting.  
### Command line arguments
- `--output $OUTPUT_FOLDER` output folder (overwrite the output folder in the config file)  
- `--input_folder $INPUT_FOLDER` input folder (overwrite the input folder in the config file) 
- `--save_rendering` save rendering video to `vis.mp4` in the output folder
- `--no_gt_traj` do not show ground truth trajectory
- `--imap` visualize results of iMAP*
- `--vis_input_frame` opens up a viewer to show input frames. Note: you need to download the dataset first. See the Run section below.

## Demo

Here you can run NICE-SLAM yourself on a short ScanNet sequence with 500 frames. 

First, download the demo data as below and the data is saved into the `./Datasets/Demo` folder. 
```bash
bash scripts/download_demo.sh
```
Next, run NICE-SLAM. It takes a few minutes with ~5G GPU memory.
```bash
python -W ignore run.py configs/Demo/demo.yaml
```
Finally, run the following command to visualize.
```bash
python visualizer.py configs/Demo/demo.yaml 
```

**NOTE:** This is for demonstration only, its configuration/performance may be different from our paper.


## Run

### Self-captured Apartment
Download the data as below and the data is saved into the `./Datasets/Apartment` folder. 
```bash
bash scripts/download_apartment.sh
```
Next, run NICE-SLAM:
```bash
python -W ignore run.py configs/Apartment/apartment.yaml
```

### ScanNet
Please follow the data downloading procedure on [ScanNet](http://www.scan-net.org/) website, and extract color/depth frames from the `.sens` file using this [code](https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/reader.py).

<details>
  <summary>[Directory structure of ScanNet (click to expand)]</summary>
  
  DATAROOT is `./Datasets` by default. If a sequence (`sceneXXXX_XX`) is stored in other places, please change the `input_folder` path in the config file or in the command line.

```
  DATAROOT
  └── scannet
      └── scans
          └── scene0000_00
              └── frames
                  ├── color
                  │   ├── 0.jpg
                  │   ├── 1.jpg
                  │   ├── ...
                  │   └── ...
                  ├── depth
                  │   ├── 0.png
                  │   ├── 1.png
                  │   ├── ...
                  │   └── ...
                  ├── intrinsic
                  └── pose
                      ├── 0.txt
                      ├── 1.txt
                      ├── ...
                      └── ...

```
</details>

Once the data is downloaded and set up properly, you can run NICE-SLAM:
```bash
python -W ignore run.py configs/ScanNet/scene0000.yaml
```

### Replica
Download the data as below and the data is saved into the `./Datasets/Replica` folder. Note that the Replica data is generated by the authors of iMAP, so please cite iMAP if you use the data.
```bash
bash scripts/download_replica.sh
```
and you can run NICE-SLAM:
```bash
python -W ignore run.py configs/Replica/room0.yaml
```
The mesh for evaluation is saved as `$OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply`, where the unseen regions are culled using all frames.


### TUM RGB-D
Download the data as below and the data is saved into the `./Datasets/TUM-RGBD` folder
```bash
bash scripts/download_tum.sh
```
Now run NICE-SLAM:
```bash
python -W ignore run.py configs/TUM_RGBD/freiburg1_desk.yaml
```
### Co-Fusion

First, download the dataset. This script should download and unpack the data automatically into the `./Datasets/CoFusion` folder.
```bash
bash scripts/download_cofusion.sh
```
Run NICE-SLAM:
```bash
python -W ignore run.py configs/CoFusion/room4.yaml
```

### Use your own RGB-D sequence from Kinect Azure 

<details>
  <summary>[Details (click to expand)]</summary>
      
1. Please first follow this [guide](http://www.open3d.org/docs/release/tutorial/sensor/azure_kinect.html#install-the-azure-kinect-sdk) to record a sequence and extract aligned color and depth images. (Remember to use `--align_depth_to_color` for `azure_kinect_recorder.py`)
  
  
    DATAROOT is `./Datasets` in default, if a sequence (`sceneXX`) is stored in other places, please change the "input_folder" path in the config file or in the command line.

    ```
      DATAROOT
      └── Own
          └── scene0
              ├── color
              │   ├── 00000.jpg
              │   ├── 00001.jpg
              │   ├── 00002.jpg
              │   ├── ...
              │   └── ...
              ├── config.json
              ├── depth
              │   ├── 00000.png
              │   ├── 00001.png
              │   ├── 00002.png
              │   ├── ...
              │   └── ...
              └── intrinsic.json

    ```


2. Prepare `.yaml` file based on the `configs/Own/sample.yaml`. Change the camera intrinsics in the config file based on `intrinsic.json`. You can also get the intrinsics of the depth camera via other tools such as MATLAB.
3. Specify the bound of the scene. If no ground truth camera pose is given, we construct world coordinates on the first frame. The X-axis is from left to right, Y-axis is from down to up, Z-axis is from front to back. 
4. Change the `input_folder` path and/or the `output` path in the config file or the command line.
5. Run NICE-SLAM.
```bash
python -W ignore run.py configs/Own/sample.yaml
```

**(Optional but highly Recommended)** If you don't want to specify the bound of the scene or manually change the config file. You can first run the Redwood tool in [Open3D](http://www.open3d.org/) and then run NICE-SLAM. Here we provide steps for the whole pipeline, beginning from recording Azure Kinect videos. (Ubuntu 18.04 and above is recommended.)
1. Download the Open3D repository.
```bash
bash scripts/download_open3d.sh
```
2. Record and extract frames.
```bash
# specify scene ID
sceneid=0
cd 3rdparty/Open3D-0.13.0/examples/python/reconstruction_system/
# record and save to .mkv file
python sensors/azure_kinect_recorder.py --align_depth_to_color --output scene$sceneid.mkv
# extract frames
python sensors/azure_kinect_mkv_reader.py --input  scene$sceneid.mkv --output dataset/scene$sceneid
```
3. Run reconstruction.
```bash
python run_system.py dataset/scene$sceneid/config.json --make --register --refine --integrate 
# back to main folder
cd ../../../../../
```
4. Prepare the config file.
```bash
python src/tools/prep_own_data.py --scene_folder 3rdparty/Open3D-0.13.0/examples/python/reconstruction_system/dataset/scene$sceneid --ouput_config configs/Own/scene$sceneid.yaml
```
5. Run NICE-SLAM.
```bash
python -W ignore run.py configs/Own/scene$sceneid.yaml
```
</details>

## iMAP*
We also provide our re-implementation of iMAP (iMAP*) for use. If you use the code, please cite both the original iMAP paper and NICE-SLAM.

### Usage
iMAP* shares a majority part of the code with NICE-SLAM. To run iMAP*, simply use `*_imap.yaml` in the config file and also add the argument `--imap` in the command line. For example, to run iMAP* on Replica room0:
```bash
python -W ignore run.py configs/Replica/room0_imap.yaml --imap 
```
To use our interactive visualizer:
```bash
python visualizer.py configs/Replica/room0_imap.yaml --imap 
```
To evaluate ATE:
```bash
python src/tools/eval_ate.py configs/Replica/room0_imap.yaml --imap 
```

<details>
  <summary>[<strong>Differences between iMAP* and the original iMAP</strong> (click to expand)]</summary>

#### Keyframe pose optimization during mapping
We do not optimize the selected keyframes' poses for iMAP*, because optimizing them usually leads to worse performance. One possible reason is that since their keyframes are selected globally, and many of them do not have overlapping regions especially when the scene gets larger. Overlap is a prerequisite for bundle adjustment (BA). For NICE-SLAM, we only select overlapping keyframes within a small window (local BA), which works well in all scenes. You can still turn on the keyframe pose optimization during mapping for iMAP* by enabling `BA` in the config file.

#### Active sampling
We disable the active sampling in iMAP*, because in our experiments we observe that it does not help to improve the performance while brings additional computational overhead. 

For the image active sampling, in each iteration the original iMAP uniformly samples 200 pixels in the entire image. Next, they divide this image into an 8x8 grid and calculate the probability distribution from the rendering losses. This means that if the resolution of an image is 1200x680 (Replica), only around 3 pixels are sampled to calculate the distribution for a 150x85 grid patch. This is not too much different from simple uniform sampling. Therefore, during mapping we use the same pixel sampling strategy as NICE-SLAM for iMAP*: uniform sampling, but even 4x more pixels than reported in the iMAP paper.

For the keyframe active sampling, the original iMAP requires rendering depth and color images for all keyframes to get the loss distribution, which is expensive and we again did not find it very helpful. Instead, as done in NICE-SLAM, iMAP* randomly samples keyframes from the keyframe list. We also let iMAP* optimize for 4x more iterations than NICE-SLAM, but their performance is still inferior. 

#### Keyframe selection
For fair comparison, we use the same keyframe selection method in iMAP* as in NICE-SLAM: add one keyframe to the keyframe list every 50 frames.

</details>

## Evaluation

### Average Trajectory Error
To evaluate the average trajectory error. Run the command below with the corresponding config file:
```bash
python src/tools/eval_ate.py configs/Replica/room0.yaml
```

### Reconstruction Error
To evaluate the reconstruction error, first download the ground truth Replica meshes where unseen region have been culled.
```bash
bash scripts/download_cull_replica_mesh.sh
```
Then run the command below (same for NICE-SLAM and iMAP*). The 2D metric requires rendering of 1000 depth images, which will take some time (~9 minutes). Use `-2d` to enable 2D metric. Use `-3d` to enable 3D metric.
```bash
# assign any output_folder and gt mesh you like, here is just an example
OUTPUT_FOLDER=output/Replica/room0
GT_MESH=cull_replica_mesh/room0.ply
python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d
```

We also provide code to cull the mesh given camera poses. Here we take culling of ground truth mesh of Replica room0 as an example.
```bash
python src/tools/cull_mesh.py --input_mesh Datasets/Replica/room0_mesh.ply --traj Datasets/Replica/room0/traj.txt --output_mesh cull_replica_mesh/room0.ply
```

<details>
  <summary>[For iMAP* evaluation (click to expand)]</summary>

  As discussed in many recent papers, e.g. UNISURF/VolSDF/NeuS, manual thresholding the volume density during marching cubes might be needed. Moreover, we find out there exist scaling differences, possibly because of the reason discussed in [NeuS](https://arxiv.org/abs/2106.10689). Therefore, ICP with scale is needed. You can use the [ICP tool](https://www.cloudcompare.org/doc/wiki/index.php?title=ICP) in [CloudCompare](https://www.danielgm.net/cc/) with default configuration with scaling enabled. 
</details>

## Acknowledgement
We adapted some codes from some awesome repositories including [convolutional_occupancy_networks](https://github.com/autonomousvision/convolutional_occupancy_networks), [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch), [lietorch](https://github.com/princeton-vl/lietorch), and [DIST-Renderer](https://github.com/B1ueber2y/DIST-Renderer). Thanks for making codes public available. We also thank [Edgar Sucar](https://edgarsucar.github.io/) for allowing us to make the Replica Dataset available.

## Citation

If you find our code or paper useful, please cite
```bibtex
@inproceedings{Zhu2022CVPR,
  author    = {Zhu, Zihan and Peng, Songyou and Larsson, Viktor and Xu, Weiwei and Bao, Hujun and Cui, Zhaopeng and Oswald, Martin R. and Pollefeys, Marc},
  title     = {NICE-SLAM: Neural Implicit Scalable Encoding for SLAM},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022}
}
```
## Contact
Contact [Zihan Zhu](mailto:zhuzihan2000@gmail.com) and [Songyou Peng](mailto:songyou.pp@gmail.com) for questions, comments and reporting bugs.
