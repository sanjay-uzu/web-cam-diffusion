import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch

from nodes import (
    CheckpointLoaderSimple,
    CLIPTextEncode,
    LoraLoader,
    VAEEncode,
    KSampler,
    SaveImage,
    NODE_CLASS_MAPPINGS,
    LoadImage,
    VAEDecode,
    VAELoader,
)
import cv2





def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from main import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()



class Pipeline:
    def __init__(self , model_name=None , lora_name=None , vae_name=None , steps=None , denoise=None , cfg=None , prompt=None):
        # Constructor function to initialize the class
        self.load_pipeline=True
        self.model_name=model_name
        self.lora_name=lora_name
        self.vae_name=vae_name
        self.steps=steps
        self.denoise=denoise
        self.cfg=cfg
        self.prompt=prompt
        
        with torch.inference_mode():
            self.checkpointloadersimple = CheckpointLoaderSimple()
            try:
                self.checkpointloadersimple_14 = self.checkpointloadersimple.load_checkpoint(
                    ckpt_name=self.model_name
                )

                self.loraloader = LoraLoader()
                self.loraloader_16 = self.loraloader.load_lora(
                    lora_name=self.lora_name,
                    strength_model=1,
                    strength_clip=1,
                    model=get_value_at_index(self.checkpointloadersimple_14, 0),
                    clip=get_value_at_index(self.checkpointloadersimple_14, 1),
                )

                self.cliptextencode = CLIPTextEncode()

                self.loadimage = LoadImage()
                self.loadimage_10 = self.loadimage.load_image(image="ComfyUI_00002_.png")

                self.vaeloader = VAELoader()
                self.vaeloader_15 = self.vaeloader.load_vae(
                    vae_name=self.vae_name
                )
                self.vaeencode = VAEEncode()
                self.vaeencode_12 = self.vaeencode.encode(
                    pixels=get_value_at_index(self.loadimage_10, 0),
                    vae=get_value_at_index(self.vaeloader_15, 0),
                )

                self.ksampler = KSampler()
                self.vaedecode = VAEDecode()
                self.saveimage = SaveImage()
            except:
                self.load_pipeline=False


    def update_settings(self, model_name=None , lora_name=None , vae_name=None , steps=None , denoise=None , cfg=None , prompt=None):
        if model_name is not None:
            self.model_name=model_name
        if lora_name is not None:
            self.lora_name=lora_name
        if vae_name is not None:
            self.vae_name=vae_name
        if steps is not None:
            self.steps=steps
        if denoise is not None:
            self.denoise=denoise
        if cfg is not None:
            self.cfg=cfg
        if prompt is not None:
            self.prompt=prompt
        with torch.inference_mode():
            self.checkpointloadersimple = CheckpointLoaderSimple()
            try:
                self.checkpointloadersimple_14 = self.checkpointloadersimple.load_checkpoint(
                    ckpt_name=self.model_name
                )

                self.loraloader = LoraLoader()
                self.loraloader_16 = self.loraloader.load_lora(
                    lora_name=self.lora_name,
                    strength_model=1,
                    strength_clip=1,
                    model=get_value_at_index(self.checkpointloadersimple_14, 0),
                    clip=get_value_at_index(self.checkpointloadersimple_14, 1),
                )

                self.cliptextencode = CLIPTextEncode()

                self.loadimage = LoadImage()
                self.loadimage_10 = self.loadimage.load_image(image="ComfyUI_00002_.png")

                self.vaeloader = VAELoader()
                self.vaeloader_15 = self.vaeloader.load_vae(
                    vae_name=self.vae_name
                )
                self.vaeencode = VAEEncode()
                self.vaeencode_12 = self.vaeencode.encode(
                    pixels=get_value_at_index(self.loadimage_10, 0),
                    vae=get_value_at_index(self.vaeloader_15, 0),
                )

                self.ksampler = KSampler()
                self.vaedecode = VAEDecode()
                self.saveimage = SaveImage()
                self.load_pipeline=True
            except:
                self.load_pipeline=False
    def process_frame(self, model_name=None , lora_name=None , vae_name=None , steps=None , denoise=None , cfg=None , prompt=None):

        with torch.inference_mode():
            if self.load_pipeline:
                # Member function to increment the value
                self.loadimage_10 = self.loadimage.load_image(image="test.png")

                self.cliptextencode_6 = self.cliptextencode.encode(
                    text=self.prompt,
                    clip=get_value_at_index(self.loraloader_16, 1),
                )

                self.cliptextencode_7 = self.cliptextencode.encode(
                    text="watermark, text\n", clip=get_value_at_index(self.loraloader_16, 1)
                )

                #vaeencode = VAEEncode()
                self.vaeencode_12 = self.vaeencode.encode(
                    pixels=get_value_at_index(self.loadimage_10, 0),
                    vae=get_value_at_index(self.vaeloader_15, 0),
                )
                self.ksampler_3 = self.ksampler.sample(
                    seed=random.randint(1, 2**64),
                    steps=self.steps,
                    cfg=self.cfg,
                    sampler_name="lcm",
                    scheduler="normal",
                    denoise=self.denoise,
                    model=get_value_at_index(self.loraloader_16, 0),
                    positive=get_value_at_index(self.cliptextencode_6, 0),
                    negative=get_value_at_index(self.cliptextencode_7, 0),
                    latent_image=get_value_at_index(self.vaeencode_12, 0),
                )

                self.vaedecode_8 = self.vaedecode.decode(
                    samples=get_value_at_index(self.ksampler_3, 0),
                    vae=get_value_at_index(self.vaeloader_15, 0),
                )
                code=get_value_at_index(self.vaedecode_8, 0)
                self.saveimage_9 = self.saveimage.save_images(
                    filename_prefix="ComfyUI", images= code
                )

        
                name=self.saveimage_9["ui"]["images"][0]["filename"]
                gen=cv2.imread(f"E:\\ComfyUI\\output\\{name}" ,  cv2.IMREAD_COLOR)
            else:
                gen=cv2.imread(f"E:\\ComfyUI\\input\\test.png" ,  cv2.IMREAD_COLOR)
        return gen




        


