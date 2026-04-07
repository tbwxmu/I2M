from PIL import Image, ImageOps, ImageChops
import numpy as np
def resize_image(image, image_size: tuple, border_size: int = 30, b="white"):
    width = image_size[0] - 2*border_size
    height = image_size[1] - 2*border_size
    width_ratio = width / image.width
    height_ratio = height / image.height
    if width_ratio < height_ratio:
        transformed_width = width
        transformed_height = round(width_ratio * image.height) 
    else:
        transformed_width = round(height_ratio * image.width)
        transformed_height = height
    transformed_image = image.resize((transformed_width, transformed_height), Image.LANCZOS)
    if b == "black":
        background = Image.new('L', (width, height), (0))
    if b == "white":
        background = Image.new('L', (width, height), (255))
    offset = (round((width-transformed_width)/2), round((height-transformed_height)/2))
    background.paste(transformed_image, offset)
    if b == "black":
        transformed_image = ImageOps.expand(background, border=border_size, fill='black')
    if b == "white":
        transformed_image = ImageOps.expand(background, border=border_size, fill='white')
    return transformed_image  
def crop_borders(image):
    background = Image.new(image.mode, image.size, image.getpixel((0,0)))
    difference = ImageChops.difference(image, background)
    difference = ImageChops.add(difference, difference, 2.0, -100)
    bbox = difference.getbbox()
    if bbox:
        return image.crop(bbox)
def transform_png_image(image_path):
    image_gt = Image.open(image_path).convert("RGB")
    image_gt = np.array(image_gt)
    image_gt[np.any(image_gt <= (200., 200., 200.), axis=2)] = [0., 0., 0.] 
    image_gt = (image_gt != 0)
    image_gt = np.array(image_gt, dtype=np.float32)
    image_gt = Image.fromarray((image_gt*255).astype(np.uint8)).convert('RGB')
    return image_gt
def transform_tif_image(tif_image_path):
    image_gt = Image.open(tif_image_path)
    image_gt = (np.array(image_gt)*255).astype(np.uint8)
    image_gt = np.repeat(image_gt[:, :, np.newaxis], 3, axis=2)
    image_gt[np.any(image_gt <= (254., 254., 254.), axis=2)] = [0., 0., 0.] 
    image_gt = (image_gt != 0)
    image_gt = np.array(image_gt, dtype=np.float32)
    image_gt = Image.fromarray((image_gt*255).astype(np.uint8)).convert('RGB')
    image_gt = crop_borders(image_gt)
    image_gt = ImageOps.expand(image_gt, border=1, fill=(255, 255, 255))
    return image_gt
def pad_image(image, border_size: int = 30, b="white"):
    width, height = image.size[0], image.size[1]
    if b == "black":
        background = Image.new('L', (width, height), (0))
    if b == "white":
        background = Image.new('L', (width, height), (255))
    offset = (0, 0)
    background.paste(image, offset)
    if b == "black":
        transformed_image = ImageOps.expand(background, border=border_size, fill='black')
    if b == "white":
        transformed_image = ImageOps.expand(background, border=border_size, fill='white')
    return transformed_image