# This is not a part of the IMDP shielding repo, but is a demonstration of more advanced coding applications than regression

import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import sys
import os
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import h5py

from nflows.flows import Flow
from nflows.transforms import AffineCouplingTransform, OneByOneConvolution
from nflows import transforms, flows

import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42) # Use the same number every time


class MultiScaleLayer(nn.Module):
    # this network takes in an image and uses invertible convolutions and down-sampling to output a smaller image
    def __init__(self, in_channels, height, width, context_dim, hidden_dim=16):
        super().__init__()

        # Make sure H and W are equal and powers of 2
        assert height == width, "Height and width must be equal"
        assert (height & (height - 1) == 0) and height > 0, "Height must be a power of 2"

        self.in_channels = in_channels
        self.height = height
        self.width = width

        # 1. Spatial Squeeze: [B, C, H, W] -> [B, 4C, H/2, W/2]
        self.squeeze = transforms.SqueezeTransform(factor=2)
        c_mid = in_channels * 4

        # 2. Invertible 1x1 Convolution (feature-wise)
        self.mixer = transforms.OneByOneConvolution(c_mid)

        # 3. Channel-wise Affine coupling
        # mask = torch.arange(c_mid) % 2  # simple alternating mask
        mask = create_checkerboard_mask(self.in_channels, self.height, self.width)
        self.register_buffer("mask", mask.float())

        mask_downsample = (torch.arange(c_mid) % 4 == 0)  # keep 1/4 info
        self.register_buffer("mask_downsample", mask_downsample.float())

        self.coupling_1 = transforms.coupling.AffineCouplingTransform(
            mask=self.mask,
            transform_net_create_fn=lambda in_d, out_d: SpatialNet(in_d, out_d, self.height//2, self.width//2,
                                                                   context_dim, hidden_dim=hidden_dim)
        )

        self.coupling_2 = transforms.coupling.AffineCouplingTransform(
            mask=1-self.mask,
            transform_net_create_fn=lambda in_d, out_d: SpatialNet(in_d, out_d, self.height//2, self.width//2,
                                                                   context_dim, hidden_dim=hidden_dim)
        )

    def forward(self, x, context=None):
        # convolve for spatial/orientation context with alternating masks
        B, C, H, W = x.shape
        x = x.view(B, -1)  # [B, C*H*W]
        x, _ = self.coupling_1.forward(x, context=context)
        x, _ = self.coupling_2.forward(x, context=context)
        x = x.view(B, -1, self.height, self.width)  # [B, C, H, W]

        # squeeze then do 1x1 convolution to mix channels
        x, _ = self.squeeze.forward(x)  # [B, 4C, H/2, W/2]
        x, _ = self.mixer.forward(x)  # [B, 4C, H/2, W/2]

        # Split channels:
        x_pass, x_split = x[:, self.mask_downsample.bool(), ...], x[:, ~(self.mask_downsample.bool()), ...]
        return x_pass, x_split

    def inverse(self, x_pass, x_split, context=None):
        # Recombine channels
        # 1. Initialize the combined tensor
        # x_pass has shape [B, C, H//2, W//2]
        # x_split has shape [B, 3C, H//2, W//2]
        B, C_pass, H, W = x_pass.shape
        C_split = x_split.shape[1]
        total_channels = C_pass + C_split

        # Create empty container on the correct device
        x = torch.zeros((B, total_channels, H, W), device=x_pass.device)

        # 2. Scatter the channels back based on the mask
        x[:, self.mask_downsample.bool(), ...] = x_pass
        x[:, (~self.mask_downsample.bool()), ...] = x_split

        # Invert squeeze and 1x1
        x, _ = self.mixer.inverse(x)  # [B, 4C, H/2, W/2]
        x, _ = self.squeeze.inverse(x)  # [B, C, H, W]

        # invert spatial convolution
        x = x.view(B, -1)  # [B, C*H*W]
        x, _ = self.coupling_2.inverse(x, context=context)
        x, _ = self.coupling_1.inverse(x, context=context)
        x = x.view(B, -1, self.height, self.width)  # [B, C, H, W]

        return x


class ChannelBottleneck(nn.Module):
    # this takes in an image, reshapes it to a vector through squeezes and convolutions,
    # then projects to a low dimensional output vector
    def __init__(self, in_c, in_d, target_dim):
        super().__init__()
        self.transforms = nn.ModuleList()

        curr_c, curr_d = in_c, in_d

        # 1. Spatial Reduction Phase: Keep the 'geometry' alive
        # We squeeze until the spatial resolution is very small (e.g., 2x2)
        while curr_d > 2 and (curr_c * curr_d * curr_d) > target_dim * 2:
            # Squeeze: [C, H, W] -> [4C, H/2, W/2]
            self.transforms.append(transforms.SqueezeTransform())
            curr_c *= 4
            curr_d //= 2

            # Mix channels so the 'important' features aren't stuck in one corner
            self.transforms.append(transforms.OneByOneConvolution(curr_c))

        # 2. Channel Reduction Phase: Flatten and split
        self.final_spatial_dim = curr_d
        self.flat_dim = curr_c * curr_d * curr_d
        self.final_dim = target_dim

        # Use an LU-Linear layer on the flattened vector to align 'target_dim'
        # features into the first indices of the vector
        if self.flat_dim > target_dim:
            self.final_mixer = transforms.CompositeTransform([
                transforms.RandomPermutation(features=self.flat_dim),
                transforms.LULinear(features=self.flat_dim)
            ])

    def forward(self, x, context=None):
        # Pass through spatial flow
        for transform in self.transforms:
            x, _ = transform.forward(x)

        # Flatten
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)

        # Final mixing to align features for splitting
        if self.flat_dim > self.final_dim:
            x, _ = self.final_mixer.forward(x, context=context)

            # Split: Keep the first 'target_dim' elements as your Image Code
            img_code = x[:, :self.final_dim]
            split_info = x[:, self.final_dim:]
            return img_code, split_info

        return x, None

    def inverse(self, img_code, split_generator=None, context=None):
        # Recombine
        if split_generator is not None and (self.flat_dim > self.final_dim):
            split_info = split_generator(img_code, context=context)
            x = torch.cat([img_code, split_info], dim=1)
            x, _ = self.final_mixer.inverse(x)
        else:
            x = img_code

        # Un-flatten
        x = x.reshape(-1, self.flat_dim // (self.final_spatial_dim ** 2),
                      self.final_spatial_dim, self.final_spatial_dim)

        # Reverse spatial flow
        for transform in reversed(self.transforms):
            x, _ = transform.inverse(x)
        return x


class SpatialNet(nn.Module):
    # this takes in a large vector representation of an image, reshapes it to C x H x W and performs convolution
    def __init__(self, in_features, out_features, h, w, context_dim, hidden_dim=16):
        super().__init__()
        self.h, self.w = h, w
        self.c = in_features // (h * w)
        # Calculate channels based on the 1D feature count nflows provides
        in_c = context_dim + self.c
        out_c = out_features // (h * w)

        self.net = nn.Sequential(
            nn.Conv2d(in_c, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_c, kernel_size=3, padding=1)
        )

    def forward(self, x, context=None):
        # x: [Batch, In_Features] (e.g., [B, 4096])
        batch_size = x.shape[0]

        # 1. Reshape to 4D for Convolution
        x = x.view(batch_size, self.c, self.h, self.w)

        if context is not None:
            ctx = context.view(batch_size, -1, 1, 1).expand(-1, -1, self.h, self.w)
            x = torch.cat([x, ctx], dim=1)

        # 2. Apply Spatial Convolutions
        x = self.net(x)

        # 3. Flatten back to 1D for nflows
        return x.reshape(batch_size, -1)


def create_checkerboard_mask(c, h, w, inverted=False):
    # Create coordinate grids
    # [H, 1] + [1, W] -> [H, W]
    x = torch.arange(h).view(-1, 1)
    y = torch.arange(w).view(1, -1)

    # Calculate (i + j) % 2
    mask = (x + y) % 2

    if inverted:
        mask = 1 - mask

    mask = mask.expand(c, h, w)
    # Reshape to [1, 1, H, W] to match (B, C, H, W) expectations
    # nflows and most coupling layers expect a mask that can broadcast
    return mask.reshape(-1).bool()


class OrientationExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim=64):
        super().__init__(observation_space, features_dim)

        c, h, w = observation_space["image"].shape

        self.levels = nn.ModuleList()

        min_res = 8  # this shouldn't be hard coded, but for this system it is fine
        while h > min_res:
            level = MultiScaleLayer(c, h, w, 0, hidden_dim=64)
            self.levels.append(level)
            c, h, w = c, h // 2, w // 2  # 4x channels, then split out 3/4th of them to keep channels the same

        self.psi_DS = ChannelBottleneck(c, h, 16)

        self.flattened_dim = self.psi_DS.final_dim
        vector_dim = observation_space["vector"].shape[0]
        self.joint_dim = self.flattened_dim + vector_dim

        self.mlp_combined = nn.Sequential(
            nn.Linear(self.joint_dim, features_dim),
            nn.ReLU()
        )

        self._features_dim = features_dim

    def forward(self, observations):

        # MultiScale layers Forward
        x = observations["image"]
        for level in self.levels:
            x, x_split = level.forward(x)

        # x is now c x min_res x min_res
        # continue down-sampling to make it similar in size to the state vector and of the same shape
        img_features, bottleneck_splits = self.psi_DS.forward(x)

        vec = observations["vector"]
        combined = torch.cat([img_features, vec], dim=1)
        return self.mlp_combined(combined)


class PreTrainDataCollector:
    def __init__(self, filename, img_shape=(64, 64, 1), vec_dim=3, label_dim=2, buffer_size=100, v_min=np.array([0.0, 0.0, 0.0]),
                 v_max=np.array([10.0, 10.0, 1.0])):
        self.file = h5py.File(filename, 'w')

        self.buffer_size = buffer_size
        self.buffer = {'images': [], 'vectors': [], 'labels': []}

        # Create datasets with 'chunks' to allow efficient writing/reading
        self.images = self.file.create_dataset("images", (0, *img_shape),
                                               maxshape=(None, *img_shape),
                                               dtype='uint8', chunks=(1, *img_shape),
                                               compression='gzip',
                                               compression_opts=4)  # lossless compression to save disk space
        self.vectors = self.file.create_dataset("vectors", (0, vec_dim),
                                                maxshape=(None, vec_dim),
                                                dtype='float32', chunks=(1, vec_dim))
        self.labels = self.file.create_dataset("labels", (0, label_dim),
                                               maxshape=(None, label_dim),
                                               dtype='float32', chunks=(1, label_dim))

        self.v_min = v_min
        self.v_max = v_max

    def add_step(self, obs, label):

        # pre-process data for encoder training
        img = obs["image"]
        img_ = img * 255
        img_ = np.clip(img_, 0, 255)
        img_ = np.round(img_).astype(np.uint8)

        # Normalize vector inputs Min-Max scaling to [0, 1]
        vec = obs["vector"]
        vec_ = (vec - self.v_min) / (self.v_max - self.v_min)

        # 1. Add to RAM buffer
        self.buffer['images'].append(img_)
        self.buffer['vectors'].append(vec_)
        self.buffer['labels'].append(label)

        # 2. If buffer is full, "Flush" to disk and empty RAM
        if len(self.buffer['images']) >= self.buffer_size:
            self.flush_to_disk()

    def flush_to_disk(self):
        curr_len = self.images.shape[0]
        new_len = curr_len + len(self.buffer['images'])

        # Resize datasets
        self.images.resize(new_len, axis=0)
        self.vectors.resize(new_len, axis=0)
        self.labels.resize(new_len, axis=0)

        # Write the whole block at once (High efficiency)
        self.images[curr_len:new_len] = np.array(self.buffer['images'])
        self.vectors[curr_len:new_len] = np.array(self.buffer['vectors'])
        self.labels[curr_len:new_len] = np.array(self.buffer['labels'])

        # CLEAR THE RAM
        self.buffer = {'images': [], 'vectors': [], 'labels': []}
        self.file.flush()  # Ensure it's physically written to disk

    def close(self):
        self.file.close()


def collect_pretrain_data(env, filename, obs_dims, v_min, v_max, n_samples=50000):

    collector = PreTrainDataCollector(filename, img_shape=(1, env.pixel_width, env.pixel_width), vec_dim=obs_dims,
                                      label_dim=4, v_min=v_min, v_max=v_max)

    images, vectors, labels = [], [], []
    obs, _ = env.reset()

    for _ in range(n_samples):
        # Random action just to move around
        action = env.action_space.sample()
        obs, _, done, timeout, _ = env.step(action)
        if done or timeout:
            obs, _ = env.reset()

        x, y = env.state[0], env.state[1]

        # distance to goal
        dist_goal = np.sqrt((x - env.goal[0]) ** 2 +
                            (y - env.goal[1]) ** 2) / env.max_dist

        # distance to nearest obstacle
        dist_obs = 9e9
        if env.obstacles:
            dist_obs = min(np.sqrt((x - o[0]) ** 2 + (y - o[1]) ** 2)
                           for o in env.obstacles) / env.max_dist

        dist_wall = min([min([10 - x, x]), min([10 - y, y])]) / env.max_dist
        dist_obs = min([dist_obs, dist_wall])

        images.append(obs["image"])
        vectors.append(obs["vector"])
        # labels are orientation and distance to the nearest obstacle and distance to goal
        label = [np.cos(env.state[2]), np.sin(env.state[2]), dist_obs, dist_goal]
        labels.append(label)

        training_obs = env.get_encoder_obs()
        collector.add_step(training_obs, label)

    if len(collector.buffer["images"]) > 0:
        collector.flush_to_disk()
    collector.close()

    return (torch.tensor(np.array(images)),
            torch.tensor(np.array(vectors), dtype=torch.float32),
            torch.tensor(np.array(labels), dtype=torch.float32))


def pretrain_extractor(extractor, images, vectors, labels,
                       epochs=20, batch_size=256, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                       test=True):
    # the goal is to have the CNN know how to extract useful information before learning a control policy
    # Small prediction head on top of extractor output
    extractor.to(device)

    pred_head = nn.Sequential(nn.Linear(extractor._features_dim, 4))  # predicts [cos, sin, dist_goal, dist_obs]
    pred_head.to(device)

    optimizer = optim.Adam(
        list(extractor.parameters()) + list(pred_head.parameters()),
        lr=1e-3
    )
    loss_fn = nn.MSELoss()

    dataset = TensorDataset(images, vectors, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for img_b, vec_b, label_b in loader:
            optimizer.zero_grad()

            # Forward through your combined extractor
            features = extractor({"image": img_b.to(device), "vector": vec_b.to(device)})
            preds = pred_head(features)

            loss = loss_fn(preds, label_b.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}  loss: {total_loss / len(loader):.4f}")

    # this check should really be done on a different data set
    if test:
        extractor.to('cpu')
        pred_head.to('cpu')
        min_err = 180
        max_err = -180
        avg_err = 0

        max_dist_err = [-9e9, -9e9]
        avg_dist_err = [0, 0]
        min_dist_err = [9e9, 9e9]
        for img_b, vec_b, label_b in loader:

            # Forward through your combined extractor
            features = extractor({"image": img_b, "vector": vec_b})
            preds = pred_head(features)
            angles = preds[:, :2]
            angles_norm = torch.nn.functional.normalize(angles, p=2, dim=1)
            angles_norm = angles_norm.detach().numpy()

            dists = preds[:, 2:]
            dists = dists.detach().numpy()

            labels = label_b.detach().numpy()

            dot_vec = angles_norm * labels[:, :2]  # A dot B = |A||B|cos(theta), both vecs have unit norm
            cos_err = np.sum(dot_vec, axis=1)  # this is cos(delta theta) then
            cos_err = np.clip(cos_err, -1.0, 1.0)  # clip shouldn't do anything, but just to ensure math stability

            delta_theta = np.arccos(cos_err)*180/np.pi

            min_err = min(min_err, min(delta_theta))
            max_err = max(max_err, max(delta_theta))
            avg_err += np.mean(np.abs(delta_theta))

            dist_err = dists - labels[:, 2:]
            for k in [0, 1]:
                min_dist_err[k] = min(min_dist_err[k], min(dist_err[k]))
                max_dist_err[k] = max(max_dist_err[k], max(dist_err[k]))
                avg_dist_err[k] += np.mean(np.abs(dist_err[k]))

        print(f"Errors (deg): {[min_err, avg_err/len(loader), max_err]}")
        for k in [0, 1]:
            print(f"Errors (dists {k}): {[min_dist_err[k], avg_dist_err[k]/len(loader), max_dist_err[k]]}")

    return extractor.to(device)


class ContinuousGridEnv(gym.Env):
    # simple single integrator with a top-down, greyscale view of the environment.
    def __init__(self, grid_size=10):
        super().__init__()
        self.grid_size = grid_size  # world is [0, grid_size] x [0, grid_size]
        self.current_step = 0
        self.max_steps = 200
        self.dt = 0.5
        self.safe = True
        self.in_goal = False

        self.max_dist = np.sqrt(2*(grid_size**2))

        # Fixed positions for testing stability

        self.num_obs = 2
        obstacles = [
            (3.5, 4.5), (3.5, 5.5), (3.5, 6.5),
            (5, 4.5), (6, 4.5), (7, 4.5)
        ]
        self.obstacles = obstacles
        self.obs_rad = 0.5

        goal = (7.5, 7.5)
        self.goal = goal
        self.goal_rad = 1.0

        self.move_regions = False
        self.episode = 0
        self.num_resample = 250
        self.num_envs = 10
        self.env_bank = [{"obs": obstacles, "goal": goal}]

        self.noise = False

        self.pixel_width = 64
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=1, shape=(1, self.pixel_width, self.pixel_width), dtype=np.float32),
            "vector": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        })

        self.action_space = spaces.Box(low=np.array([-2.0]), high=np.array([2.0]), dtype=np.float32)

        self.reset()

    def set_state(self, x0):
        self.state[0] = x0[0]
        self.state[1] = x0[1]
        self.state[2] = x0[2]

    def randomize_regions(self):
        # can move obstacles and goals during training for generalization

        obs_list = []
        r = self.obs_rad
        obs_spacing = 2*r
        x_min, x_max = 0, self.grid_size
        y_min, y_max = 0, self.grid_size

        for k in range(self.num_obs):
            if np.random.randn(1) > 0:  # vertical
                x_center = np.random.uniform(x_min + r, x_max - r)
                y_center = np.random.uniform(y_min + r + obs_spacing, y_max - r - obs_spacing)

                obs_list.extend([(x_center, y_center - obs_spacing),
                                 (x_center, y_center),
                                 (x_center, y_center + obs_spacing)])
            else:
                x_center = np.random.uniform(x_min + r + obs_spacing, x_max - r - obs_spacing)
                y_center = np.random.uniform(y_min + r, y_max - r)

                obs_list.extend([(x_center - obs_spacing, y_center),
                                 (x_center, y_center),
                                 (x_center + obs_spacing, y_center)])

        self.obstacles = obs_list

        max_tries = 25
        r_goal = self.goal_rad
        goal = (np.random.uniform(x_min + r_goal, x_max - r_goal),
                np.random.uniform(y_min + r_goal, y_max - r_goal))
        for _ in range(max_tries):
            if all(np.max(tuple(np.abs(a - b) for a, b in zip(goal, o))) >= (r_goal + r) for o in self.obstacles):
                break  # goal doesn't overlap
            else:
                goal = (np.random.uniform(x_min + r_goal, x_max - r_goal),
                        np.random.uniform(y_min + r_goal, y_max - r_goal))

        return obs_list, goal

    def sample_environments(self):
        for _ in range(self.num_envs):
            obs, goal = self.randomize_regions()
            self.env_bank.append({"obs": obs, "goal": goal})

    def reset(self, seed=None, options=None):
        # State: [x, y, theta]
        super().reset(seed=seed)

        if self.move_regions and (self.episode % self.num_resample == 0):
            env_num = np.random.choice(len(self.env_bank))
            env_info = self.env_bank[env_num]
            self.obstacles = env_info["obs"]
            self.goal = env_info["goal"]

        while True:
            # don't initialize in an obstacle
            self.state = np.array([
                np.random.uniform(0, self.grid_size),
                np.random.uniform(0, self.grid_size),
                np.random.uniform(-np.pi, np.pi)
            ])
            safe = True
            if self.obstacles:
                for o in self.obstacles:
                    if np.abs(self.state[0] - o[0]) <= self.obs_rad and np.abs(self.state[1] - o[1]) <= self.obs_rad:
                        safe = False
                        break
            if safe:
                break
        self.safe = True
        self.in_goal = False
        self.current_step = 0
        self.episode += 1
        return self.get_obs(), {}

    def step(self, angular_rate):
        """
        action_angle: a float representing the direction to move.
        The agent moves a distance of ~1.0 in that direction.
        """
        #
        goal_x_dir = self.goal[0] - self.state[0]
        goal_y_dir = self.goal[1] - self.state[1]
        dist_0 = np.sqrt(goal_x_dir**2 + goal_y_dir**2)/self.max_dist  # normalized distance to goal

        # Update orientation by the angular rate
        self.state[2] = self.state[2] + angular_rate[0]*self.dt
        if self.noise:
            self.state[2] += np.random.uniform(-0.1, 0.1)

        self.state[2] = (self.state[2] + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]

        # Calculate new velocity vector, it's a unit vector velocity with a step size of dt
        dx = np.cos(self.state[2])*self.dt
        dy = np.sin(self.state[2])*self.dt

        if self.noise:
            # add noise to dynamics for generalization/ smoothness of policy
            noise_mag = np.random.uniform(0.0, 0.3)
            noise_dir = np.random.uniform(0.0, 2.0*np.pi)
            dx += noise_mag*np.cos(noise_dir)*self.dt
            dy += noise_mag*np.sin(noise_dir)*self.dt

        # Update position
        self.state[0] = self.state[0] + dx
        self.state[1] = self.state[1] + dy

        # calculate reward, punish for being close to obstacles or walls, reward for progress to goal
        r_obs = 0
        if self.obstacles:
            for o in self.obstacles:
                if np.abs(self.state[0] - o[0]) <= self.obs_rad and np.abs(self.state[1] - o[1]) <= self.obs_rad:
                    self.safe = False
                    break

                dist_to_obs = np.sqrt((self.state[0] - o[0])**2 + (self.state[1] - o[1])**2)
                if dist_to_obs < 0.8:
                    r_obs -= 0.2 * np.exp(-(dist_to_obs - self.obs_rad))

        dist_to_edge_top = np.sqrt((self.state[0] - 10)**2 + (self.state[1] - 10)**2)
        r_obs -= 0.2 * np.exp(-(dist_to_edge_top - 0.1))

        dist_to_edge_bottom = np.sqrt((self.state[0] - 0)**2 + (self.state[1] - 0)**2)
        r_obs -= 0.2 * np.exp(-(dist_to_edge_bottom - 0.1))

        if (self.state[0] <= 0.1) or (self.state[0] >= (self.grid_size - 0.1)) or \
                (self.state[1] <= 0.1) or (self.state[1] >= (self.grid_size - 0.1)):
            self.safe = False

        if np.abs(self.state[0] - self.goal[0]) <= self.goal_rad and np.abs(self.state[1] - self.goal[1]) <= self.goal_rad:
            self.in_goal = True

        obs_1 = self.get_obs()

        goal_x_dir = self.goal[0] - self.state[0]
        goal_y_dir = self.goal[1] - self.state[1]
        dist_1 = np.sqrt(goal_x_dir**2 + goal_y_dir**2)/self.max_dist  # normalized distance to goal

        done = False
        # punish for turning aggressively
        turn_weight = 2.0
        if self.safe:
            if self.in_goal:
                done = True
                reward = 50.
            else:
                reward = -0.5 + 30*(dist_0 - dist_1) - turn_weight*np.sqrt((angular_rate[0])**2)  # progress based reward and time based penalty
        else:
            reward = -50.
            done = True

        self.current_step += 1
        timeout = self.current_step >= self.max_steps
        if timeout and not self.in_goal:
            reward = -30

        # penalty for being near an obstacle
        reward += r_obs

        return obs_1, reward, done, timeout, {}

    def get_obs(self):
        # return direction to goal and image rendering
        x, y, theta = self.state
        goal_x_dir = self.goal[0] - x
        goal_y_dir = self.goal[1] - y

        norm_ = np.sqrt(goal_x_dir**2 + goal_y_dir**2) + 1e-6
        inputs = np.array([goal_x_dir/norm_, goal_y_dir/norm_])

        vec = torch.tensor(inputs, dtype=torch.float32)

        img = self.render_image()
        return {"image": img.numpy(), "vector": vec.numpy()}

    def get_encoder_obs(self):
        # Return state as [x, y] for the encoder context
        x, y, theta = self.state

        inputs = np.array([x, y])

        vec = torch.tensor(inputs, dtype=torch.float32)

        img = self.render_image()
        return {"image": img.numpy(), "vector": vec.numpy()}

    def render_image(self):
        res = self.pixel_width
        # 1. Create Coordinate Grids
        y_coords, x_coords = np.ogrid[:res, :res]

        # Normalize pixel coordinates to match grid_size [0, 10], add a buffer for visibility at edges
        px = (x_coords / res) * (self.grid_size + 1.5) - 0.75
        py = (y_coords / res) * (self.grid_size + 1.5) - 0.75

        # fixed texture
        grid = 0.2 * (np.sin(px) + np.cos(py)) - 0.1

        # 3. Add Obstacles (Repulsion/Dark Blobs)
        # We use a Gaussian: exp(-dist^2 / sigma)
        if self.obstacles:
            for ox, oy in self.obstacles:
                dist_sq = (px - ox) ** 2 + (py - oy) ** 2
                grid -= 1.0 * np.exp(-dist_sq / 1.0)  # sigma=1

        # 4. Add the Goal (Attraction/Bright Blob)
        gx, gy = self.goal
        dist_sq_goal = (px - gx) ** 2 + (py - gy) ** 2
        grid += 0.5 * np.exp(-dist_sq_goal / 1.5)

        def blend_layer(grid, mask, target_value):
            return (1.0 - mask) * grid + (mask * target_value)

        dist_sq_robot = (px - self.state[0]) ** 2 + (py - self.state[1]) ** 2
        body_mask = np.exp(-dist_sq_robot / 0.15)  # > 0.5

        # also add a small point to give orientation
        dist_ = 1.5
        dist_sq_robot = (px - (self.state[0] + dist_*self.dt*np.cos(self.state[2]))) ** 2 + (py - (self.state[1] + dist_*self.dt*np.sin(self.state[2]))) ** 2
        front_mask = np.exp(-dist_sq_robot / 0.03)   # > 0.5

        grid = blend_layer(grid, body_mask, 2.0)
        grid = blend_layer(grid, front_mask, 2.0)

        # Clipping and Normalization
        grid = np.clip(grid, -1, 1)
        grid = (grid + 1) / 2.0  # Map to [0, 1]

        return torch.tensor(grid, dtype=torch.float32).unsqueeze(0)  # 1 x H x W


def simulate_ppo(model, env, sim_time=60.0, x0=None):
    path_x = []
    path_y = []
    imgs = []

    steps = int(sim_time / env.dt)

    # Robot initial state
    env.reset()

    if x0 is not None:
        env.set_state(np.array(x0))

    for idx in range(steps):
        path_x.append(env.state[0])
        path_y.append(env.state[1])

        obs = env.get_obs()
        imgs.append(obs["image"])
        # figure out control from state
        action = model.predict(obs, deterministic=True)

        # Step robot
        _, _, _, _, _ = env.step(action[0])
        safe = env.safe
        in_goal = env.in_goal

        # Stop if reached
        if in_goal:
            steps = idx + 1
            print(f"Goal reached at step {steps}, time {steps * env.dt:.2f}s. Position {env.state[0]:.3f}, {env.state[1]:.3f}")
            break
        if not safe:
            steps = idx + 1
            print(f"Violates safety at step {steps}, time {steps * env.dt:.2f}s. Position {env.state[0]:.3f}, {env.state[1]:.3f}")
            break

    path_x.append(env.state[0])
    path_y.append(env.state[1])
    obs = env.get_obs()
    imgs.append(obs["image"])
    final_pos = np.array([env.state[0], env.state[1], env.state[2]])

    return [path_x, path_y], final_pos, imgs


def plot_filled_square(ax, region, radius, color='cyan', alpha=0.6):
    """
        Plots a filled square on a given 2D axes.

        Args:
            ax (matplotlib.axes.Axes): The 3D axes object to plot on.
            region: tuple of region center
            radius: radius or region (squares though)
            color (str): The color of the cube's faces.
            alpha (float): The transparency of the cube's faces (0.0 to 1.0).
        """

    xmin, xmax = region[0] - radius, region[0] + radius
    ymin, ymax = region[1] - radius, region[1] + radius

    x_corners = [xmin, xmax, xmax, xmin, xmin]
    y_corners = [ymin, ymin, ymax, ymax, ymin]
    ax.fill(x_corners, y_corners, color=color, alpha=alpha, linewidth=0.1, edgecolor='k')


def plot_observation(image, x, plt_dir, title="Agent Observation"):
    """
    image: torch.Tensor of shape [1, H, W]
    state: torch.Tensor [x, y, sin_theta, cos_theta, dx_goal, dy_goal]
    """
    # img_np = image.squeeze().cpu().numpy()
    img_np = image.squeeze()

    fig, ax = plt.subplots()

    # 1. Plot the Rendered Image
    im = ax.imshow(img_np, cmap='magma', origin='lower')
    ax.set_title(f"{title}\n(Neural Input)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)
    plt.savefig(plt_dir + f'/image_{x}.png', dpi=300)
    plt.close()


def plot_simulation_ppo(model, env, plt_dir, sim_time=60.0, x0=None):

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)
    ax.set_title("Single Integrator path with image")

    # Plot obstacles
    if env.obstacles:
        for obs in env.obstacles:
            plot_filled_square(ax, obs, env.obs_rad, color='tab:gray', alpha=0.8)

    plot_filled_square(ax, env.goal, env.goal_rad, color='darkgreen', alpha=0.8)

    for init in x0:
        path, final_pos, imgs = simulate_ppo(model, env, sim_time=sim_time, x0=init)
        path_x = path[0]
        path_y = path[1]

        # Plot path
        ax.plot(path_x, path_y, '-k', linewidth=2)  # , label='robot path'

        # Plot start and goal
        ax.plot(path_x[0], path_y[0], 'go')  # , label='start'

        # Plot final robot pose
        fx, fy, fth = final_pos[0], final_pos[1], final_pos[2]
        ax.plot(fx, fy, 'bo')  # , label='final pose'
        # draw heading arrow
        ah = 0.6
        ax.arrow(fx, fy, ah * np.cos(fth), ah * np.sin(fth), head_width=0.15, head_length=0.2, color='b')

    ax.legend()
    plt.show(block=False)
    plt.pause(0.001)
    plt.savefig(plt_dir + f'/single_int_nn.png', dpi=300)
    plt.close()

    for k in range(len(path_x)):
        plot_observation(imgs[k], k, plt_dir)


if __name__ == "__main__":
    env = ContinuousGridEnv()

    exp_dir = os.path.dirname(os.path.abspath(__file__))
    plt_dir = exp_dir + "/test_fig"
    if not os.path.isdir(plt_dir):
        os.mkdir(plt_dir)
    train = int(sys.argv[1])

    if train:

        env.noise = True
        env.move_regions = False

        policy_kwargs = dict(
            features_extractor_class=OrientationExtractor,
            features_extractor_kwargs=dict(features_dim=16),
        )

        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=5e-3,
            gamma=0.95,
            n_steps=2048,
            batch_size=256,
            ent_coef=0.1,
            verbose=1,
            tensorboard_log="./tb_logs/"
        )

        # pre-train CNN feature extractor, it must identify orientation and distances
        obs_dims = 2
        v_min = np.array([0.0, 0.0])
        v_max = np.array([10.0, 10.0])
        data = collect_pretrain_data(env, "GridWorldPreTrain.h5", obs_dims, v_min, v_max, n_samples=100000)
        extractor = model.policy.features_extractor
        extractor = pretrain_extractor(extractor, data[0], data[1], data[2], epochs=20, batch_size=512, test=False)

        model.policy.features_extractor = extractor

        # freeze cnn weights
        for param in model.policy.features_extractor.psi_DS.parameters():
            param.requires_grad = False

        for param in model.policy.features_extractor.levels.parameters():
            param.requires_grad = False

        try:
            model.learn(total_timesteps=100000, tb_log_name="ppo_run_1")
            model.save("gridworld_controller")
        except KeyboardInterrupt:
            print("Training interrupted, saving model...")
            model.save("gridworld_ppo_interrupted")
    else:
        env.noise = False
        model = PPO.load("gridworld_controller.zip", env=env)
        x0 = [[5,  6, -1.02737234], [3, 8, np.pi/2], [9.5, 9.5, 7*np.pi/8], [1.0, 3.0, -np.pi/4.0],
              [1, 1, 0], [8, 1, np.pi], [4, 2, -np.pi/2], [5, 3, np.pi/2],
              [1, 9, -np.pi], [5, 9, np.pi/2]]
        plot_simulation_ppo(model, env, plt_dir, sim_time=20.0, x0=x0)

    env.close()
