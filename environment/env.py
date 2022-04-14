import random
import time
import os
import warnings

import gym
from gym import spaces
from gym.utils import seeding
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as transforms
import torch
import carla
from queue import Queue
import queue

from config import INPUT_DIM, MIN_STEERING, MAX_STEERING, JERK_REWARD_WEIGHT, MAX_STEERING_DIFF

class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __del__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


class Env(gym.Env):
    def __init__(self, client, vae=None, min_throttle=0.4, max_throttle=0.6, n_command_history=20, frame_skip=1, n_stack=1, action_lambda=0.5, episode_length=800):
        self.client = client
        self.world = self.client.get_world()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            synchronous_mode=True,
            fixed_delta_seconds=0.05))
        self.episode_length = episode_length
        
        # settings = self.world.get_settings()
        # settings.fixed_delta_seconds=None
        # self.world.apply_settings(settings)

        self.blueprint_library = self.world.get_blueprint_library()
        self.sync_mode = None
        self.actor_list = []

        # save last n commands
        self.n_commands = 2
        self.n_command_history = n_command_history
        self.command_history = np.zeros((1, self.n_commands * n_command_history))
        self.n_stack = n_stack
        self.stacked_obs = None

        # assumes that we are using VAE input
        self.vae = vae
        self.z_size = None
        if vae is not None:
            self.z_size = vae.zdim
        else:
            print("NO VAE")
        
        self.observation_space = spaces.Box(low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(1, self.z_size + self.n_commands * n_command_history),
            dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-MAX_STEERING, -1]),
            high=np.array([MAX_STEERING, 1]),
            dtype=np.float32)

        self.min_throttle = min_throttle
        self.max_throttle = max_throttle
        self.frame_skip = frame_skip
        self.action_lambda = action_lambda
        self.last_throttle = 0.0
        self.seed()

    def jerk_penalty(self):
        """
        Add a continuity penalty to limit jerk.
        :return: (float)
        """
        jerk_penalty = 0
        if self.n_command_history > 1:
            # Take only last command into account
            for i in range(1):
                steering = self.command_history[0, -2 * (i + 1)]
                prev_steering = self.command_history[0, -2 * (i + 2)]
                steering_diff = (prev_steering - steering) / (MAX_STEERING - MIN_STEERING)

                if abs(steering_diff) > MAX_STEERING_DIFF:
                    error = abs(steering_diff) - MAX_STEERING_DIFF
                    jerk_penalty += JERK_REWARD_WEIGHT * (error ** 2)
                else:
                    jerk_penalty += 0
        return jerk_penalty

    def postprocessing_step(self, action, observation, reward, done, info):
        """
        Update the reward (add jerk_penalty if needed), the command history
        and stack new observation (when using frame-stacking).

        :param action: ([float])
        :param observation: (np.ndarray)
        :param reward: (float)
        :param done: (bool)
        :param info: (dict)
        :return: (np.ndarray, float, bool, dict)
        """
        # Update command history
        if self.n_command_history > 0:
            self.command_history = np.roll(self.command_history, shift=-self.n_commands, axis=-1)
            self.command_history[..., -self.n_commands:] = action
            observation = np.concatenate((observation, self.command_history), axis=-1)

        jerk_penalty = self.jerk_penalty()
        # Cancel reward if the continuity constrain is violated
        if jerk_penalty > 0 and reward > 0:
            reward = 0
        reward -= jerk_penalty

        if self.n_stack > 1:
            self.stacked_obs = np.roll(self.stacked_obs, shift=-observation.shape[-1], axis=-1)
            if done:
                self.stacked_obs[...] = 0
            self.stacked_obs[..., -observation.shape[-1]:] = observation
            return self.stacked_obs, reward, done, info

        # print("postprocessing_step finished")

        return observation, reward, done, info

    def step(self, action):
        # self.world.tick()

        self.num_step += 1
        # Convert from [-1, 1] to [0, 1]
        t = (action[1] + 1) / 2
        # Convert from [0, 1] to [min, max]
        action[1] = (1 - t) * self.min_throttle + self.max_throttle * t

        # Clip steering angle rate to enforce continuity
        if self.n_command_history > 0:
            prev_steering = self.command_history[0, -2]
            max_diff = (MAX_STEERING_DIFF - 1e-5) * (MAX_STEERING - MIN_STEERING)
            diff = np.clip(action[0] - prev_steering, -max_diff, max_diff)
            action[0] = prev_steering + diff

        # control = Control()
        # print(action)
        # print(type(action[0]), type(action[1]))
        control = carla.VehicleControl(throttle=float(action[1]), steer=float(action[0]))

        # Repeat action if using frame_skip
        # for _ in range(self.frame_skip):
        # self.client.send_control(control)
        self.vehicle.apply_control(control)
        # measurements, sensor_data = self.client.read_data()
        velocity = self.vehicle.get_velocity().length()
        
        # im = self.camera_queue.get()

        # if self.lane_queue.empty():
        #     lane_det = None
        # else:
        #     lane_det = self.lane_queue.get()
        
        # if self.collision_queue.empty():
        #     collision_det = None
        # else:
        #     collision_det = self.collision_queue.get()
        snapshot, im = self.sync_mode.tick(timeout=2.0)
                
        im = np.array(im.raw_data).reshape((800, 800, 4))
        im = im[:, :, :3] # convert to BGR
        im = preprocess_image(im)
        transform = transforms.ToTensor()
        im = transform(im)[None, :]
        with torch.no_grad():
            _, observation, _ = self.vae(im)

        observation = observation.detach().numpy()

        assert(velocity is not None)
        reward, done = self.reward(velocity, action)

        self.last_throttle = action[1]

        if self.num_step > self.episode_length:
            done = True

        if done:
            print("num_step:", self.num_step)

        return self.postprocessing_step(action, observation, reward, done, {})

    def reset(self):
        print("Start to reset env")
        self.destroy_agents()
        
        self.num_step = 0
        # vehicle
        while True:
            try:
                bp = random.choice(self.blueprint_library.filter('vehicle.*')) # randomly choose a vehicle
                transform = random.choice(self.world.get_map().get_spawn_points()) # randomly choose a start position
                self.vehicle = self.world.spawn_actor(bp, transform) # vehicle object
                self.actor_list = [self.vehicle]
                break
            except Exception as e:
                print('changing position, ',str(e))
                time.sleep(0.01)
        
        # camera
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '800')
        camera_bp.set_attribute('fov', '90')
        # camera_bp.set_attribute('sensor_tick', '0.1')
        bound_x = 0.5 + self.vehicle.bounding_box.extent.x
        bound_y = 0.5 + self.vehicle.bounding_box.extent.y
        bound_z = 0.5 + self.vehicle.bounding_box.extent.z
        camera_location = carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z)
        camera_rotation = carla.Rotation(8.0, 0.0, 0.0)
        camera_transform = carla.Transform(camera_location, camera_rotation)
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.SpringArm)
        self.actor_list.append(self.camera)
        # self.camera_queue = Queue()

        # lane detection
        lane_bp = self.blueprint_library.find('sensor.other.lane_invasion')
        lane_location = carla.Location(0,0,0)
        lane_rotation = carla.Rotation(0,0,0)
        lane_transform = carla.Transform(lane_location, lane_rotation)
        self.lane_detection = self.world.spawn_actor(lane_bp, lane_transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(self.lane_detection)
        self.lane_queue = Queue()

        # collision detection
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        collision_location = carla.Location(0,0,0)
        collision_rotation = carla.Rotation(0,0,0)
        collision_transform = carla.Transform(collision_location, collision_rotation)
        self.collision_detection = self.world.spawn_actor(collision_bp, collision_transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(self.collision_detection)
        self.collision_queue = Queue()

        observation = None

        # start listening
        # self.camera.listen(lambda image: self.camera_queue.put(image))
        self.lane_detection.listen(lambda lane: self.lane_queue.put(lane))
        self.collision_detection.listen(lambda collision: self.collision_queue.put(collision))

        # self.world.tick()

        # while self.camera_queue.empty():
        #     time.sleep(0.01)
        #     self.world.tick()

        # print(self.camera_queue.empty()) # True
        # print(self.lane_queue.empty()) # True
        # print(self.collision_queue.empty()) # True
        
        # if self.camera_queue.empty():
        #     im = np.random.rand(800,800,4).astype('float32')
            # print("MARK 5.1")
        # else:

        self.sync_mode = CarlaSyncMode(self.world, self.camera)
        snapshot, im = self.sync_mode.tick(timeout=2.0)

        im = np.array(im.raw_data).reshape((800, 800, 4))
        im = im[:, :, :3] # convert to BGR
        im = preprocess_image(im)
        transform = transforms.ToTensor()
        im = transform(im)[None, :]

        with torch.no_grad():
            _, observation, _ = self.vae(im)
        observation = observation.detach().numpy()
        self.command_history = np.zeros((1, self.n_commands * self.n_command_history))

        if self.n_command_history > 0:
            observation = np.concatenate((observation, self.command_history), axis=-1)

        if self.n_stack > 1:
            self.stacked_obs[...] = 0
            self.stacked_obs[..., -observation.shape[-1]:] = observation
            return self.stacked_obs

        print('reset finished')
        return observation


    def reward(self, velocity, action):
        """
        :param measurements:
        :return: reward, done
        """
        done = False

        """speed"""
        # # In the wayve.ai paper, speed has been used as reward
        speed_reward = 10 * velocity

        """road"""
        if not self.collision_queue.empty():
            self.collision_queue = Queue()
            self.lane_queue = Queue()
            return -100.0, True

        if not self.lane_queue.empty():
            self.collision_queue = Queue()
            self.lane_queue = Queue()
            return -10.0, True

        return speed_reward, done
    
    def destroy_agents(self):
        for actor in self.actor_list:

            # If it has a callback attached, remove it first
            if hasattr(actor, 'is_listening') and actor.is_listening:
                actor.stop()

            # If it's still alive - desstroy it
            if actor.is_alive:
                actor.destroy()

        if self.sync_mode is not None:
            del self.sync_mode
        
        self.sync_mode = None
        self.actor_list = []
        # self.camera_queue = Queue()
        self.lane_queue = Queue()
        self.collision_queue = Queue()



def preprocess_image(image, convert_to_rgb=False):
    """
    Crop, resize and normalize image.
    Optionnally it also converts the image from BGR to RGB.
    :param image: (np.ndarray) image (BGR or RGB)
    :param convert_to_rgb: (bool) whether the conversion to rgb is needed or not
    :return: (np.ndarray)
    """
    # Crop
    # Region of interest
    image = image[400:, :]
    # Resize
    im = cv2.resize(image, (160, 80), interpolation=cv2.INTER_AREA)
    # Convert BGR to RGB
    if convert_to_rgb:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    return im