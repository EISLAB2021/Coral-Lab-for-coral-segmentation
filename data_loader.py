import os
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io, transform
import random
import cv2

class SegDataLoader(Dataset):
    def __init__(self, data_dir, phase, data_size):
        self.data_dir = data_dir
        self.phase = phase
        self.data_size = data_size
        self.files = self.read_txt()

    def read_txt(self):
        image_list = []
        with open(os.path.join(self.data_dir, self.phase+'.txt'), 'r') as f:
            for file in f.readlines():
                image_list.append(file.rstrip('\n'))
        return image_list

    def __len__(self):
        return len(self.files)

    # Perform random data augmentation in each epoch to improve data diversity
    def augments(self, image, label):
        h, w,_ = image.shape

        # =================== Horizontally flip with a 50% probability =================== #
        flip_h_flag = random.choice([True, False])
        if flip_h_flag:
            image = np.flip(image, axis=1)
            label = np.flip(label, axis=1)

        # ==================== Vertically flip with a 50% probability ==================== #
        flip_v_flag = random.choice([True, False])
        if flip_v_flag:
            image = np.flip(image, axis=0)
            label = np.flip(label, axis=0)

        # =================== Brightness change with a 1/3 probability =================== #
        brightness_flag = random.choice([True, False, False])
        if brightness_flag:
            lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB) # RGB --> LAB, L is brightness
            # Generate a random L-value adjustment from a normal distribution (+-5~10) has almost no impact on RGB information
            l_adjust = np.random.normal(0, 10)  # normal distribution, N(0,10)
            l_adjust = np.clip(l_adjust, -10, 10)   # limited in [-10, 10]

            l_channel, a_channel, b_channel = cv2.split(lab_image) # extract L,A,B channels
            l_channel = l_channel + l_adjust # adjust L value to change brightness
            # Standardize variable types
            l_channel = l_channel.astype(np.float32)
            a_channel = a_channel.astype(np.float32)
            b_channel = b_channel.astype(np.float32)
            adjusted_lab_image = cv2.merge((l_channel, a_channel, b_channel)) # merge L,A,B channels

            # LAB --> RGB
            image = cv2.cvtColor(adjusted_lab_image, cv2.COLOR_LAB2RGB)

        # ============= Conditionally apply color augments  with a 50% probability ============= #
        # Calculate average of R and G channels
        image = image.astype(np.float32)
        mean_r = np.mean(image[..., 0])
        mean_g = np.mean(image[..., 1])
        # If R_aver - G_aver < 2, add color augment module (decrease R value)
        if abs(mean_r - mean_g) < 2:
            reduce_r_flag = random.choice([False, False])  # decrease R value with a 50% probability
            if reduce_r_flag:
                image[..., 0] = np.clip(image[..., 0] - 0.5, 0, 255)

        # After brightness and color adjustment, change 'float32' to 'uint8'
        image = np.clip(image, 0, 255)

        # =================== Rotate random angle with a 50% probability =================== #
        rotate_flag = random.choice([True, False])
        if rotate_flag:
            rotate_angle = random.uniform(0, 120)
            # image = transform.rotate(image, rotate_angle,mode='reflect') # shouldn't use reflect mode
            # label = transform.rotate(label, rotate_angle,mode='reflect')
            image = transform.rotate(image, rotate_angle) # keep black background
            label = transform.rotate(label, rotate_angle)

        # =================== Crop random part with a 1/3 probability =================== #
        crop_flag = random.choice([True, False, False])
        if crop_flag:
            top = int(random.uniform(0, h//4))
            left = int(random.uniform(0, w//4))
            size = int(random.uniform(h//2, h))
            image = image[top:top+size, left:left+size]
            label = label[top:top+size, left:left+size]
            # Cropped part is scaled by bicubic method
            image = cv2.resize(image,
                               dsize=(self.data_size, self.data_size),
                               interpolation=cv2.INTER_CUBIC)
            label = cv2.resize(label,
                               dsize=(self.data_size, self.data_size),
                               interpolation=cv2.INTER_CUBIC)

        return image, label

    @ staticmethod
    def image_process(image, label):
        image = image.astype(np.float32) / 255.0
        label = label.astype(np.float32) / 255.0
        h, w, _ = image.shape
        # check h, w
        if h > 512 or w > 512:
            h_ = random.choice(range(0, max(0, h - 512)))
            w_ = random.choice(range(0, max(0, w - 512)))
            image = image[h_:h_+512, w_:w_+512, :]
            label = label[h_:h_+512, w_:w_+512, :]

        return image, label

    def __getitem__(self, item):
        # ============================== Read image files ============================== #
        # image = io.imread(os.path.join(self.data_dir, 'images', self.files[item]), as_gray=True)
        # image = io.imread(os.path.join(self.data_dir, 'images', self.files[item]))
        if os.path.isdir(os.path.join(self.data_dir, 'images')):
            folder = 'images'
        elif os.path.isdir(os.path.join(self.data_dir, 'Image')):
            folder = 'Image'
        else:
            raise FileNotFoundError("Neither 'labels' nor 'Mask' folder exists in the data directory.")
        image = io.imread(os.path.join(self.data_dir, folder, self.files[item]))
        image = image[:, :, :3]

        # ============================== Read label files ============================== #
        # label = io.imread(os.path.join(self.data_dir, 'labels', self.files[item]), as_gray=True) # gray image is within [0,1]
        if os.path.isdir(os.path.join(self.data_dir, 'labels')):
            folder = 'labels'
        elif os.path.isdir(os.path.join(self.data_dir, 'Mask')):
            folder = 'Mask'
        else:
            raise FileNotFoundError("Neither 'labels' nor 'Mask' folder exists in the data directory.")
        label = io.imread(os.path.join(self.data_dir, folder, self.files[item]), as_gray=True)  # gray image is within [0,1]
        label = (label * 255).astype(np.float32)

        # ============================== Process image and label files ============================== #
        image, label = self.image_process(image, label)

        if self.phase == 'train':
            image, label = self.augments(image, label)

        # image = image - 0
        label = np.where(label >= 0.5, 1.0, 0.0)

        # image = np.expand_dims(image, -1)
        image = image.transpose(2, 0, 1)

        return torch.from_numpy(image).float(), torch.from_numpy(label).long()

class DataLoaderInference(Dataset):

    def __init__(self, data_dir, data_size=512):
        self.image_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif']
        self.data_dir = data_dir
        self.data_size = data_size

        self.files = []
        self.glob_images('')
        print('There are {} images'.format(len(self.files)))

    def glob_images(self, data_dir):
        """
        glob all images in the dirs and their all child_dirs
        :param data_dir: input data dirs
        """
        for file in os.listdir(os.path.join(self.data_dir, data_dir)):
            file_dir = os.path.join(self.data_dir, data_dir, file)
            if file_dir.find('result') != -1:
                continue
            elif os.path.isdir(file_dir):
                # finding the sub dirs
                self.glob_images(os.path.join(data_dir, file))
            elif os.path.isfile(file_dir):
                # check format
                file_format = os.path.basename(file_dir).split('.')[-1].lower()
                if file_format in self.image_formats:
                    self.files.append(os.path.join(data_dir, file))
                else:
                    continue

    def __len__(self):
        return len(self.files)

    @ staticmethod
    def move_image(image):
        # crop the center
        h, w, cha = image.shape

        if w == h:
            return image

        max_size = max(w, h) + 1
        filled_images = np.zeros((max_size, max_size, cha), dtype=np.float32)

        delta = (max_size - min(h, w)) // 2
        if h == max_size:
            filled_images[1:, delta: delta+w, cha:] = image
        else:
            filled_images[delta:delta+h, 1:,cha:] = image

        return filled_images

    def image_process(self, image):
        # image = image.astype(np.float32) / np.max(image)
        image = image.astype(np.float32) / 255.0

        # med_intensity = np.percentile(image, 50)
        # image = np.power(image, 0.6)
        # if exposure.is_low_contrast(image, 0.15, lower_percentile=10, upper_percentile=90):
        #     med_intensity = np.percentile(image,  50)
        #     if med_intensity >= 0.5:
        #         gamma = min(1 + (med_intensity - 0.5) * 8, 2.2)
        #         image = np.power(image, gamma)
        #     else:
        #         gamma = max(med_intensity * 2.2, 1/2.2)
        #         image = np.power(image, gamma)

        image = self.move_image(image)

        image = cv2.resize(image, (self.data_size, self.data_size), interpolation=cv2.INTER_CUBIC)
        return image

    def __getitem__(self, item):
        # basename = os.path.basename(self.files[item])

        image = io.imread(os.path.join(self.data_dir, self.files[item]))
        # image = io.imread(os.path.join(self.data_dir, 'images', self.files[item]), as_gray=True)

        image = image[:, :, :3]
        # print(self.files[item])



        image = self.image_process(image)
        image = image - 0

        # image = np.expand_dims(image, -1)
        image = image.transpose(2, 0, 1)

        return self.files[item], torch.from_numpy(image).float()

# ===================== this module is not used in this project ===================== #
class ClsDataLoader(Dataset):
    def __init__(self, data_dir, phase, data_size):
        self.data_dir = data_dir
        self.phase = phase
        self.data_size = data_size
        self.files = self.read_txt()

    def read_txt(self):
        image_list = []
        with open(os.path.join(self.data_dir, self.phase+'.txt'), 'r') as f:
            for file in f.readlines():
                image_list.append(file.rstrip('\n'))
        return image_list

    def __len__(self):
        return len(self.files)

    def augments(self, image):
        flip_h_flag = random.choice([True, False])
        if flip_h_flag:
            image = np.flip(image, axis=1)

        flip_v_flag = random.choice([True, False])
        if flip_v_flag:
            image = np.flip(image, axis=0)

        rotate_flag = random.choice([True, False])
        if rotate_flag:
            rotate_angle = random.uniform(0, 180)
            image = transform.rotate(image, rotate_angle)

        return image

    def __getitem__(self, item):
        # image = io.imread(os.path.join(self.data_dir, 'images', self.files[item]), as_gray=True)
        image = io.imread(os.path.join(self.data_dir, 'images', self.files[item]))
        image = image.astype(np.float32) / 255.0

        # label is shown in filename string with index of 0
        label = int(self.files[item][0])

        if self.phase == 'train':
            image = self.auguments(image)

        image = image - 0.5
        image = np.expand_dims(image, -1)
        image = image.transpose(2, 0, 1)

        return torch.from_numpy(image).float(), label
# ===================== this module is not used in this project ===================== #



