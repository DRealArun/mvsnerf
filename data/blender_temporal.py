import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
import cv2
from PIL import Image
from torchvision import transforms as T
from scipy.spatial.transform import Rotation as ROT

from .ray_utils import *


class BlenderTemporalDataset(Dataset):
    def __init__(self, args, split='train', load_ref=False):
        self.args = args
        self.root_dir = args.datadir
        self.anno_file_path = os.path.join(self.root_dir, "annotation.json")
        self.split = split
        self.img_wh = (512, 512)  # w,h (resolution after downsampling)
        self.downsample_wh = (1920/self.img_wh[0], 1080/self.img_wh[1])
        self.define_transforms()

        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        if not load_ref:
            self.read_meta()

        self.white_back = True
    
    def _get_frame_n_camera_ids(self, file_idx):
        cam_str_prefix = "camera_"
        camera_id = cam_str_prefix + file_idx.split(cam_str_prefix)[-1].split('_')[0]
        frame_id = int(file_idx.split('_')[-1])
        return camera_id, frame_id

    def _get_rgb_image_name(self, file_idx):
        return 'RGB_' + file_idx + '.jpg'

    def _get_depth_map_name(self, file_idx):
        return 'Depth_' + file_idx + '.exr'

    def _clip_depth_map(self, depth):
        valid = (depth > self.near) & (depth < self.far)
        depth = np.where(valid, depth, np.zeros_like(depth))
        return depth

    def load_image(self, image_path):
        if '.jpg' in image_path:
            pil_img = Image.open(image_path).convert('RGB')
            image = np.asarray(pil_img)
        elif '.exr' in image_path:
            image = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            image = image[:, :, 1]
            image = self._clip_depth_map(image)
        return image

    def _get_valid_path(self, file_name):
        file_path = os.path.join(self.root_dir, file_name)
        assert os.path.exists(file_path), "File {} does not exist. Aborting...".format(file_path)
        return file_path
    
    def _get_camera_location(self, cam_id, meta):
        loc_dict = meta[cam_id]['location']
        loc = np.array([float(loc_dict['x']), float(loc_dict['y']), float(loc_dict['z'])])
        return loc

    def _get_camera_orientation(self, cam_id, meta):
        ori_dict = meta[cam_id]['orientation']
        ori = np.array([float(ori_dict['x']), float(ori_dict['y']), float(ori_dict['z']), float(ori_dict['w'])])
        return ori

    def _get_camera_intrinsic_matrix(self, cam_id, meta):
        _K = np.array(meta[cam_id]['K'])
        _K[0, :] /= 1920
        _K[1, :] /= 1080
        K = np.block(
            [[_K, np.zeros((3, 1))],
             [np.zeros((1, 3)), 1]]).astype(np.float32)
        return K

    def read_meta(self):
        assert os.path.exists(self.anno_file_path), "Annotation file missing!"
        with open(self.anno_file_path, 'r') as f:
            self.meta = json.load(f)

        self.ids = []
        self.num_frames = 0
        self.num_cameras = len(self.meta.keys())
        for root, dirs, files in os.walk(self.root_dir, topdown=False):
            for file in files:
                file_name = file.split('/')[-1]
                if 'RGB' in file:
                    file_idx = file_name.split("RGB_")[-1].split('.')[0]
                    frame_num = int(file_idx.split('camera_')[-1].split('_')[-1])
                    self.num_frames = max(self.num_frames, frame_num)
                    if file_idx not in self.ids:
                        self.ids.append(file_idx)

        def _get_sorting_value(idx):
            cid, fid = self._get_frame_n_camera_ids(idx)
            cid = int(cid.split("camera_")[-1])
            fid = int(fid) * self.num_cameras
            return cid + fid

        self.ids.sort(key=_get_sorting_value)

        # sub select training views
        train_split = int(70*(len(self.ids)//self.num_cameras)/100)*self.num_cameras
        # val_split = len(self.ids)-train_split
        if self.split == 'train':
            self.meta['frames'] = self.ids[0:train_split]
        else:
            self.meta['frames'] = self.ids[train_split:]
        print(f'===> {self.split}ing index: {self.ids}')

        intrinsic_mat = self._get_camera_intrinsic_matrix('camera_1', self.meta)
        w, h = self.img_wh
        intrinsic_mat[0, :] *= w
        intrinsic_mat[1, :] *= h
        self.focal_x = intrinsic_mat[0, 0]
        self.focal_y = intrinsic_mat[1, 1]

        # bounds, common for all scenes #3.5 to 10 works well
        self.near = 2.0 
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal_x, self.focal_y])  # (h, w, 3)


        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        for frame in self.meta['frames']:
            cam_id, _ = self._get_frame_n_camera_ids(frame)
            orientation = self._get_camera_orientation(cam_id, self.meta)
            translation = self._get_camera_location(cam_id, self.meta)
            R_mat = ROT.from_quat(orientation).as_matrix()
            transformation_mat = np.eye(4)
            transformation_mat[0:3,0:3]= R_mat
            transformation_mat[0:3,3] = translation[0:3]
            pose = np.array(transformation_mat) @ self.blender2opencv
            self.poses += [pose]
            c2w = torch.FloatTensor(pose)

            file_name = self._get_rgb_image_name(frame)
            image_path = self._get_valid_path(file_name)
            depth_file_name = self._get_depth_map_name(frame)
            depth_path = self._get_valid_path(depth_file_name)

            self.image_paths += [image_path]

            img = self.load_image(image_path)
            dep = self.load_image(depth_path)
            # Merge and combine before converting to tensor
            orig_h, orig_w = np.shape(img)[0:2]
            img = np.where(dep.reshape(orig_h, orig_w, 1) > 0, img, np.ones_like(img)*255)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)
            img = self.transform(img)  # (3, h, w)
            img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGBA

            dep_resized = cv2.resize(dep, (w, h), interpolation=cv2.INTER_LANCZOS4)
            dep_resized = torch.from_numpy(dep_resized)
            dep_resized = dep_resized.view(1, -1).permute(1, 0)
            self.all_masks += [dep_resized > 0]
            self.all_rgbs += [img]

            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)

            self.all_rays += [torch.cat([rays_o, rays_d,
                                         self.near * torch.ones_like(rays_o[:, :1]),
                                         self.far * torch.ones_like(rays_o[:, :1])],
                                        1)]  # (h*w, 8)
            self.all_masks += []

        self.poses = np.stack(self.poses)
        if 'train' == self.split:
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
            self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)

    def read_source_views(self, file=f"transforms_train.json", pair_idx=None, device=torch.device("cpu")):
        assert os.path.exists(self.anno_file_path), "Annotation file missing!"
        with open(self.anno_file_path, 'r') as f:
            meta = json.load(f)

        intrinsic_mat = self._get_camera_intrinsic_matrix('camera_1', meta)
        w, h = self.img_wh
        intrinsic_mat[0, :] *= w
        intrinsic_mat[1, :] *= h
        focal_x = intrinsic_mat[0, 0]
        focal_y = intrinsic_mat[1, 1]


        src_transform = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        ids = []
        num_frames = 0
        num_cameras = len(meta.keys())
        for root, dirs, files in os.walk(self.root_dir, topdown=False):
            for file in files:
                file_name = file.split('/')[-1]
                if 'RGB' in file:
                    file_idx = file_name.split("RGB_")[-1].split('.')[0]
                    frame_num = int(file_idx.split('camera_')[-1].split('_')[-1])
                    num_frames = max(num_frames, frame_num)
                    if file_idx not in ids:
                        ids.append(file_idx)

        def _get_sorting_value(idx):
            cid, fid = self._get_frame_n_camera_ids(idx)
            cid = int(cid.split("camera_")[-1])
            fid = int(fid) * num_cameras
            return cid + fid

        ids.sort(key=_get_sorting_value)

        # sub select training views
        train_split = int(70*(len(ids)//num_cameras)/100)*num_cameras
        meta['frames'] = ids[0:train_split]
        print(f'====> ref idx: {ids}')

        imgs, proj_mats = [], []
        intrinsics, c2ws, w2cs = [],[],[]
        for i,frame in enumerate(meta['frames']):
            cam_id, _ = self._get_frame_n_camera_ids(frame)
            orientation = self._get_camera_orientation(cam_id, meta)
            translation = self._get_camera_location(cam_id, meta)
            R_mat = ROT.from_quat(orientation).as_matrix()
            transformation_mat = np.eye(4)
            transformation_mat[0:3,0:3]= R_mat
            transformation_mat[0:3,3] = translation[0:3]
            c2w = np.array(transformation_mat) @ self.blender2opencv
            w2c = np.linalg.inv(c2w)
            c2ws.append(c2w)
            w2cs.append(w2c)

            # build proj mat from source views to ref view
            proj_mat_l = np.eye(4)
            intrinsic = np.array([[focal_x, 0, w / 2], [0, focal_y, h / 2], [0, 0, 1]])
            intrinsics.append(intrinsic.copy())
            intrinsic[:2] = intrinsic[:2] / 4  # 4 times downscale in the feature space
            proj_mat_l[:3, :4] = intrinsic @ w2c[:3, :4]
            if i == 0:  # reference view
                ref_proj_inv = np.linalg.inv(proj_mat_l)
                proj_mats += [np.eye(4)]
            else:
                proj_mats += [proj_mat_l @ ref_proj_inv]


            file_name = self._get_rgb_image_name(frame)
            image_path = self._get_valid_path(file_name)
            depth_file_name = self._get_depth_map_name(frame)
            depth_path = self._get_valid_path(depth_file_name)

            img = self.load_image(image_path)
            dep = self.load_image(depth_path)

            # Merge and combine before converting to tensor
            orig_h, orig_w = np.shape(img)[0:2]
            img = np.where(dep.reshape(orig_h, orig_w, 1) > 0, img, np.ones_like(img)*255)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)
            img = self.transform(img)  # (3, h, w)
            imgs.append(src_transform(img))

        pose_source = {}
        pose_source['c2ws'] = torch.from_numpy(np.stack(c2ws)).float().to(device)
        pose_source['w2cs'] = torch.from_numpy(np.stack(w2cs)).float().to(device)
        pose_source['intrinsics'] = torch.from_numpy(np.stack(intrinsics)).float().to(device)

        near_far_source = [2.0,6.0]
        imgs = torch.stack(imgs).float().unsqueeze(0).to(device)
        proj_mats = torch.from_numpy(np.stack(proj_mats)[:,:3]).float().unsqueeze(0).to(device)
        return imgs, proj_mats, near_far_source, pose_source

    def load_poses_all(self, file=f"transforms_train.json"):
        assert os.path.exists(self.anno_file_path), "Annotation file missing!"
        with open(self.anno_file_path, 'r') as f:
            meta = json.load(f)

        c2ws = []
        for i,frame in enumerate(meta['frames']):
            cam_id, _ = self._get_frame_n_camera_ids(frame)
            orientation = self._get_camera_orientation(cam_id, meta)
            translation = self._get_camera_location(cam_id, meta)
            R_mat = ROT.from_quat(orientation).as_matrix()
            transformation_mat = np.eye(4)
            transformation_mat[0:3,0:3]= R_mat
            transformation_mat[0:3,3] = translation[0:3]
            c2ws.append(np.array(transformation_mat) @ self.blender2opencv)
        return np.stack(c2ws)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        if self.split == 'train':  # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else:  # create data for each image separately
            # frame = self.meta['frames'][idx]
            # c2w = torch.FloatTensor(frame['transform_matrix']) @ self.blender2opencv

            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            mask = self.all_masks[idx] # for quantity evaluation

            sample = {'rays': rays,
                      'rgbs': img,
                      'mask': mask}
        return sample