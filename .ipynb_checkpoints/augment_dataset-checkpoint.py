# %%time
import PIL
import imgaug.augmenters as iaa
from datasets import Image, Dataset, load_dataset
import numpy as np

def get_image(image_file):    
    image = PIL.Image.open(image_file).convert('RGB')
    return image

def augment_img(seq, image):
    # Augment image from image arr
    aug_img = seq.augment(image=np.array(image))
    
    # convert from arr to RGB 
    aug_img = PIL.Image.fromarray(aug_img)
    
    return aug_img

def get_augimg(seq, image_file):
    # Open Image file path with PIL
    image = get_image(image_file)
    
    # Augment image
    aug_img = augment_img(seq, image)
    return aug_img


def df_to_dataset(df, augment=False):
    # Transform DF to Dataset
    dataset = Dataset.from_pandas(df, preserve_index = False)
        
    # Image Augmentation
    if augment:
        seq = iaa.Sequential([
            iaa.Crop(px=(0, 32)), # crop images from each side by 0 to 32px (randomly chosen)
            iaa.Sometimes(0.5, # 50% Probability
                          iaa.GaussianBlur(sigma=(0, 2.0)) # blur images with a sigma of 0 to 3.0
                         ),
            iaa.Sometimes(0.5,
                          iaa.AdditiveGaussianNoise(scale=(0, 0.2*255))
                         ),
        ])
        
        def transforms(examples):
            examples["image"] = [augment_img(seq, image) for image in examples["image"]]
            return examples
        
        dataset.set_transform(transforms)
        
        # dataset = dataset.map(lambda example: {"image": get_augimg(seq, example["image"])}, num_proc=8)

    # Convert to Dataset Image class
    dataset = dataset.cast_column("image", Image())

    return dataset