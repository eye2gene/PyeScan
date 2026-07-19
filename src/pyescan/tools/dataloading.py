 #TODO: Remove dependence on cv2 by using PIL Image transform
from PIL import Image
import numpy as np
import cv2

# this is the resulotion the model will work with 
# default axial resolution for Heidelberg
DEFAULT_RESOLUTION_OCT = 0.01158, 0.0038716 # Lateral, Axial
DEFAULT_RESOLUTION_FAF = 0.031542, 0.031542 # Width, Height

class SegmentationDataLoader():

    def __init__(self, out_size=(512, 512), modality="OCT"):
        
        self.patch_size = out_size
        self.scale_factor = 1.
        
        if modality == "OCT":
            self.targ_resolutions_mm_width = DEFAULT_RESOLUTION_OCT[0]
            self.targ_resolutions_mm_height = DEFAULT_RESOLUTION_OCT[1]
            
            # augmentation parameters
            self.max_scale = 1.1
            self.max_ty = 64
            self.max_tx = 32
            self.max_rotation = 5 # 1 degree (larger rotations can lead to unrealistic shadow but could be useful still)
            
        elif modality == "FAF":
            self.targ_resolutions_mm_width = DEFAULT_RESOLUTION_FAF[0]
            self.targ_resolutions_mm_height = DEFAULT_RESOLUTION_FAF[1]
            
            # augmentation parameters
            self.max_scale = 1.
            self.max_ty = 0
            self.max_tx = 0
            self.max_rotation = 0
        
        self.brightness_range = (-0.1, 0.3)
        self.contrast_range = (0.5, 1.1)
        self.gamma_range = (0.7, 1.5)

    def _get_transform_matrix(self, in_size, out_size, rotate, scale, translate):
        # center to top left corner
        h, w, *_ = in_size
        cy = h / 2
        cx = w / 2
        C1 = np.array([
            [1, 0, -cx],
            [0, 1, -cy],
            [0, 0, 1]
        ], dtype=np.float) 

        # rotate
        th = rotate * np.pi / 180
        R = np.array([
            [np.cos(th), -np.sin(th), 0],
            [np.sin(th), np.cos(th), 0],
            [0, 0, 1]
        ], dtype=np.float)

        # scale
        sy, sx = scale
        S = np.array([
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, 1]
        ], dtype=np.float) 

        # top left corner to center
        h, w = out_size
        ty = h / 2
        tx = w / 2
        C2 = np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ], dtype=np.float) 

        # translate
        ty, tx = translate
        T = np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ], dtype=np.float) 

        return T @ C2 @ S @ R @ C1

    def _get_image_patch(self, image, labels, patch_size, rotate=0., scale=1., translate=(0.,0.)):

        if not hasattr(scale, '__len__'): scale = (scale, scale)

        M = self._get_transform_matrix(image.shape, patch_size, rotate, scale, translate)
        x = cv2.warpAffine(image, M[:2], dsize=patch_size) / 255

        #M_inv = np.linalg.inv(M)
        #image = Image.fromarray(image)
        #np.array(image.transform(patch_size, Image.AFFINE, data=M_inv.flatten()[:6], resample=Image.NEAREST)) / 255

        if labels is None:
            y = None
        else:
            y = cv2.warpAffine(labels, M[:2], dsize=patch_size, flags=cv2.INTER_NEAREST)
        return x, y

    def _intensity_transform(self, image, brightness=0., contrast=1., gamma=1.):
        return np.clip((contrast * image ** gamma + brightness), 0, 1)
    
    def _process_sample(self, raw_image, raw_labels, augment=False, image_scaling_factor=(1.,1.)):
                        
        if augment:
            rotation = np.random.uniform(-self.max_rotation, self.max_rotation)
            scale = np.random.uniform(1, self.max_scale)
            if np.random.choice([True, False]):
                scale = 1 / scale
            ty = np.random.uniform(-self.max_ty, self.max_ty)
            tx = np.random.uniform(-self.max_tx, self.max_tx)
        else:
            rotation = 0
            scale = 1
            ty, tx = 0, 0
            
        scale = (scale * image_scaling_factor[0], scale * image_scaling_factor[1])
        
        image, labels = self._get_image_patch(raw_image, raw_labels, self.patch_size, rotation, scale, (tx, ty))

        if augment:
            if np.random.choice([True, False]):
                image = image[:,::-1]
                if not labels is None:
                    labels = labels[:,::-1]
                    
            brightness = np.random.uniform(*self.brightness_range)
            contrast = np.random.uniform(*self.contrast_range)
            gamma = np.random.uniform(*self.gamma_range)

            image = self._intensity_transform(image, brightness, contrast, gamma)
        return image, labels
    
    def _get_scale_factor(self, scan_resolutions_mm_width, scan_resolutions_mm_height):
        scale_x = scan_resolutions_mm_width / self.targ_resolutions_mm_width * self.scale_factor  
        scale_y = scan_resolutions_mm_height / self.targ_resolutions_mm_height * self.scale_factor
        return scale_x, scale_y
    
    def _load_image(self, image_path, labels_paths, image_shape):
        try:
            raw_scan = np.array(Image.open(image_path).convert('L'))
        except:
            raise Exception("Failed to load ", image_path)
            
        if not labels_paths: return raw_scan, None

        w, h, *_ = image_shape
        raw_labels = np.zeros((h, w, len(labels_paths)), dtype=np.float)

        for i, label_path in enumerate(labels_paths):
            if label_path is None or str(label_path) == "nan":
                raw_labels[...,i] = -1.
            elif label_path == "MarkedAsDone":
                raw_labels[...,i] = 0.
            else:
                try:
                    raw_labels[...,i] = (np.array(Image.open(label_path).convert('L')) > 127)
                except:
                    raise Exception("Failed to load ", feature_pth)

        return raw_scan, raw_labels 

    def _load_image_from_row(self, row, features):
        image_shape = row.size_width, row.size_height
        labels_paths = [row["file_path_{}".format(feature)] for feature in features]
        return self._load_image(row.file_path, labels_paths, image_shape)
    
    def _load_data_from_row(self, row, features, augment=False):
        raw_image, raw_labels = self._load_image_from_row(row, features)
        
        scale_factor = self._get_scale_factor(row.resolutions_mm_width, row.resolutions_mm_height)
        image, labels = self._process_sample(raw_image, raw_labels, augment, image_scaling_factor=scale_factor)
        return image, labels
        
    
def __get_default_scaling_from_row(row, scale_factor=1., target_resolution=None):
    if target_resolution is None:
        if row.modality == "OCT":
            target_resolution = DEFAULT_RESOLUTION_OCT
        else:
            target_resolution = DEFAULT_RESOLUTION_FAF
            
    scale_x = row.resolutions_mm_width / target_resolution[0] * scale_factor  
    scale_y = row.resolutions_mm_height / target_resolution[1] * scale_factor
    return (scale_x, scale_y)


def __load_processed_data_from_row(row, features, patch_size, augment=False, target_resolution=None):
                  
    # augmentation parameters
    # TODO 
    max_scale = 1.1
    max_ty = 64
    max_tx = 32
    max_rotation = 5 # 1 degree (larger rotations can lead to unrealistic shadow but could be useful still)
    
    if augment:
        rotation = np.random.uniform(-max_rotation, max_rotation)
        scale = np.random.uniform(1, max_scale)
        if np.random.choice([True, False]):
            scale = 1 / scale
        ty = np.random.uniform(-max_ty, max_ty)
        tx = np.random.uniform(-max_tx, max_tx)
    else:
        rotation = 0
        scale = 1
        ty, tx = 0, 0
        
    raw_image, raw_labels = load_raw_data_from_row(row, features)
    
    scale = get_default_scaling_from_row(row, scale, target_resolution)
    image, labels = get_patch(raw_image, raw_labels, patch_size, rotation, scale, (tx, ty))
    
    print(labels.shape)
    
    if augment:
        if np.random.choice([True, False]):
            image = image[:,::-1]
            if not labels is None:
                labels = labels[:,::-1]

        image = intensity_transform(image)
    return image, labels

def __scale_to_target_resolution(input_image, output_dimension, image_spacing, target_spacing):
    
    resolution_axial, resolution_lateral = image_spacing
    target_resolution_axial, target_resolution_lateral = target_spacing
    
    scale_x = resolution_lateral / target_resolution_lateral
    scale_y = resolution_axial / target_resolution_axial
    
    # Rescale the image
    new_width = int(image.width * scale_x)
    new_height = int(image.height * scale_y)
    resized_image = image.resize((new_width, new_height))

    # Define the crop size and centre coordinates
    crop_x = int(new_width/2 - output_dimension[0]/2)
    crop_y = int(new_height/2 - output_dimension[1]/2)

    # Crop the image
    cropped_image = resized_image.crop((crop_x, crop_y, crop_x + crop_size[0], crop_y + crop_size[1]))

    return cropped_img