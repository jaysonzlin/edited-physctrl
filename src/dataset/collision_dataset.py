import numpy as np
import torch
from torch.utils.data import Dataset
import os
import h5py

def compute_kinematics(P, Q, dt=0.0417, has_gravity=False):
    # P, Q: (N, 3) tensors
    C_P = P.mean(dim=0)
    C_Q = Q.mean(dim=0)
    
    v = (C_Q - C_P) / dt
    if has_gravity:
        # v_avg = v_0 + 0.5 * a * dt. Thus v_0 = v_avg - 0.5 * a * dt
        # where a = -9.81
        v[2] += 0.5 * 9.81 * dt
        
    P_c = P - C_P
    Q_c = Q - C_Q
    
    H = P_c.T @ Q_c
    U, S, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T
    if torch.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
        
    w_x = (R[2, 1] - R[1, 2]) / (2 * dt)
    w_y = (R[0, 2] - R[2, 0]) / (2 * dt)
    w_z = (R[1, 0] - R[0, 1]) / (2 * dt)
    w = torch.tensor([w_x, w_y, w_z], dtype=P.dtype, device=P.device)
    
    return v, w

class CollisionDataset(Dataset):
    def __init__(self, split, cfg):
        self.cfg = cfg
        self.split = split
        if split == 'val':
            self.dataset_path = cfg.get('val_dataset_path', cfg.dataset_path)
        else:
            self.dataset_path = cfg.dataset_path
        self.mode = cfg.mode # 'ae' or 'diff'
        self.n_frames_interval = cfg.n_frames_interval
        self.n_training_frames = cfg.n_training_frames
        
        # Load all h5 files in the dataset path (no train/test split, all 110 datapoints)
        if os.path.exists(self.dataset_path):
            self.split_lst = [f for f in sorted(os.listdir(self.dataset_path)) if f.endswith('.h5')]
        else:
            self.split_lst = []
            
        print('Number of data:', len(self.split_lst))
        
        self.datapoints = []
        for m in self.split_lst:
            self.datapoints.append({"datapoint": m, "start_idx": 0})
            self.datapoints.append({"datapoint": m, "start_idx": 24})
            self.datapoints.append({"datapoint": m, "start_idx": 48})

    def __getitem__(self, index):
        if self.mode == 'ae':
            return self.get_collision_ae(index)
        elif self.mode == 'diff':
            return self.get_collision_diff(index)
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")

    def __len__(self):
        if self.mode == 'ae':
            # Setup for future ae mode implementation
            return 0 
        elif self.mode == 'diff':
            return len(self.datapoints)
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")
            
    def get_collision_ae(self, index):
        raise NotImplementedError("ae mode not implemented yet")
    
    def get_collision_diff(self, index):
        datapoint = self.datapoints[index]
        datapoint_name = datapoint["datapoint"]
        start_idx = datapoint["start_idx"]

        datapoint_info = {}
        datapoint_info["datapoint"] = datapoint_name
        
        datapoint_data = {}
        datapoint_data['datapoint'] = datapoint_name
        
        datapoint_info["indices"] = np.arange(start_idx, start_idx + self.n_training_frames)

        # Load the h5 file
        file_path = os.path.join(self.dataset_path, datapoint_name)
        with h5py.File(file_path, 'r') as datapoint_metas:
            # The keys explicitly required to be loaded under the same names
            keys_to_load = [
                'floor_height', 'gravity',
                'diameter1', 'friction1', 'p1', 'rho1', 'v1', 'w1',
                'diameter2', 'friction2', 'p2', 'rho2', 'v2', 'w2',
            ]
            
            for key in keys_to_load:
                if key in datapoint_metas:
                    data = np.array(datapoint_metas[key])
                    # Convert the loaded arrays to torch tensors
                    if key == 'gravity':
                        datapoint_data[key] = torch.from_numpy(data).bool()
                    else:
                        tensor_data = torch.from_numpy(data).float()
                        if key in ['v1', 'v2', 'w1', 'w2']:
                            tensor_data = tensor_data / 2.0
                        elif key in ['p1', 'p2', 'floor_height']:
                            tensor_data = tensor_data / 5.0
                        if key in ['rho1', 'rho2', 'friction1', 'friction2']:
                            tensor_data = tensor_data.view(-1)
                        datapoint_data[key] = tensor_data

            if 'x' in datapoint_metas:
                x_raw = torch.from_numpy(np.array(datapoint_metas['x']))
                # Prepare x by grabbing the init_pc and the target frames based on start_idx
                raw_start_idx = start_idx * self.n_frames_interval
                raw_end_idx = raw_start_idx + self.n_training_frames * self.n_frames_interval + 1
                
                points_src = x_raw[raw_start_idx : raw_start_idx + 1]
                points_tgt = x_raw[raw_start_idx + self.n_frames_interval : raw_end_idx : self.n_frames_interval]
                datapoint_data['points_src'] = points_src.float() / 5.0
                datapoint_data['points_tgt'] = points_tgt.float() / 5.0
                
                if start_idx > 0:
                    has_gravity = False
                    if 'gravity' in datapoint_data and datapoint_data['gravity'].item() == True:
                        has_gravity = True
                        
                    P1 = x_raw[raw_start_idx, :2048]
                    Q1 = x_raw[raw_start_idx + 1, :2048]
                    v1_dyn, w1_dyn = compute_kinematics(P1, Q1, has_gravity=has_gravity)
                    datapoint_data['v1'] = v1_dyn.float() / 2.0
                    datapoint_data['w1'] = w1_dyn.float() / 2.0
                    
                    P2 = x_raw[raw_start_idx, 2048:]
                    Q2 = x_raw[raw_start_idx + 1, 2048:]
                    v2_dyn, w2_dyn = compute_kinematics(P2, Q2, has_gravity=has_gravity)
                    datapoint_data['v2'] = v2_dyn.float() / 2.0
                    datapoint_data['w2'] = w2_dyn.float() / 2.0

        return datapoint_data, datapoint_info