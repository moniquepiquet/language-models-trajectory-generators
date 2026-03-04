from google import genai
from google.genai import types

import json
import textwrap
import time
import os
import dataclasses
from io import BytesIO
from PIL import Image
import base64
import numpy as np
import matplotlib.pyplot as plt
import torch

@dataclasses.dataclass(frozen=True)
class SegmentationMask:
  # bounding box pixel coordinates (not normalized)
  y0: int  # in [0..height - 1]
  x0: int  # in [0..width - 1]
  y1: int  # in [0..height - 1]
  x1: int  # in [0..width - 1]
  mask: np.array  # [img_height, img_width] with values 0..255
  label: str

def parse_json(json_output):
  # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            # Remove everything before "```json"
            json_output = "\n".join(lines[i + 1 :])
            # Remove everything after the closing "```"
            json_output = json_output.split("```")[0]
            break  # Exit the loop once "```json" is found
    return json_output
  
def call_gemini_robotics_er(img, prompt, settings=None):

    google_api_key = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=google_api_key)
    MODEL_ID = "gemini-robotics-er-1.5-preview"

    default_config = types.GenerateContentConfig(
        temperature=0.5,
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )

    if settings is None:
        settings = default_config

    image_response = client.models.generate_content(
          model=MODEL_ID,
          contents=[img, prompt],
          config=settings,
    )

    print(image_response.text)
    return parse_json(image_response.text)


def parse_segmentation_masks(
    predicted_str: str, *, img_height: int, img_width: int
) -> list[SegmentationMask]:
  items = json.loads(parse_json(predicted_str))
  masks = []
  for item in items:
    raw_box = item["box_2d"]
    abs_y0 = int(item["box_2d"][0] / 1000 * img_height)
    abs_x0 = int(item["box_2d"][1] / 1000 * img_width)
    abs_y1 = int(item["box_2d"][2] / 1000 * img_height)
    abs_x1 = int(item["box_2d"][3] / 1000 * img_width)
    if abs_y0 >= abs_y1 or abs_x0 >= abs_x1:
      print("Invalid bounding box", item["box_2d"])
      continue
    label = item["label"]
    png_str = item["mask"]
    if not png_str.startswith("data:image/png;base64,"):
      print("Invalid mask")
      continue
    png_str = png_str.removeprefix("data:image/png;base64,")
    png_str = base64.b64decode(png_str)
    mask = Image.open(BytesIO(png_str))
    bbox_height = abs_y1 - abs_y0
    bbox_width = abs_x1 - abs_x0
    if bbox_height < 1 or bbox_width < 1:
      print("Invalid bounding box")
      continue
    mask = mask.resize(
        (bbox_width, bbox_height), resample=Image.Resampling.BILINEAR
    )
    np_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    np_mask[abs_y0:abs_y1, abs_x0:abs_x1] = mask
    masks.append(
        SegmentationMask(abs_y0, abs_x0, abs_y1, abs_x1, np_mask, label)
    )
  return masks

def parse_masks_to_tensor(json_output, img_height, img_width):
    try:
        data = json.loads(json_output)
        if not data:
            return torch.zeros((0, img_height, img_width), dtype=torch.bool)

        mask_arrays = []

        for item in data:
            # --- 1. Get and Denormalize Bounding Box ---
            # The model returns [ymin, xmin, ymax, xmax] normalized to 0-1000
            box = item["box_2d"]

            # Convert to pixels
            ymin = int(box[0] * img_height / 1000)
            xmin = int(box[1] * img_width / 1000)
            ymax = int(box[2] * img_height / 1000)
            xmax = int(box[3] * img_width / 1000)

            # Ensure coordinates are within image bounds
            ymin, xmin = max(0, ymin), max(0, xmin)
            ymax, xmax = min(img_height, ymax), min(img_width, xmax)

            # Calculate box dimensions
            box_h = ymax - ymin
            box_w = xmax - xmin

            # Handle degenerate boxes (width or height is 0)
            if box_h <= 0 or box_w <= 0:
                mask_arrays.append(np.zeros((img_height, img_width), dtype=np.uint8))
                continue

            # --- 2. Process the Mask ---
            b64_string = item["mask"]
            if "data:image" in b64_string:
                b64_string = b64_string.split(",")[1]

            mask_bytes = base64.b64decode(b64_string)
            mask_crop = Image.open(BytesIO(mask_bytes))

            # Resize the mask crop to match the BOX dimensions, not the image dimensions
            mask_crop = mask_crop.resize((box_w, box_h), resample=Image.NEAREST)

            # Convert crop to numpy
            mask_crop_np = np.array(mask_crop)

            # --- 3. Paste into Full Canvas ---
            # Create a black canvas of the full image size
            full_mask = np.zeros((img_height, img_width), dtype=np.uint8)

            # Insert the cropped mask into the correct position
            # (Check if mask is 2D or 3D, sometimes PIL opens as RGB)
            if len(mask_crop_np.shape) == 3:
                # If RGB, take just one channel or convert to grayscale
                mask_crop_np = mask_crop_np[:, :, 0]

            full_mask[ymin:ymax, xmin:xmax] = (mask_crop_np > 0).astype(np.uint8)

            mask_arrays.append(full_mask)

        # Stack and convert to Boolean Tensor (required for draw_segmentation_masks)
        if not mask_arrays:
             return torch.zeros((0, img_height, img_width), dtype=torch.bool)

        tensor_np = np.stack(mask_arrays, axis=0)
        return torch.from_numpy(tensor_np).bool() # Return boolean directly

    except Exception as e:
        print(f"Parsing error: {e}")
        return torch.zeros((0, img_height, img_width), dtype=torch.bool)
        #print(f"Parsing error: {e}")
        #return torch.zeros((0, img_height, img_width), dtype=torch.uint8)


def get_gemini_output(image, segmentation_texts):
    phrases = segmentation_texts
    segmentation_texts = ", ".join(segmentation_texts) #queries
    width, height = image.size

    prompt = textwrap.dedent("""\
    Provide the segmentation masks for the following objects in this image: %s.

    The answer should follow the JSON format:
    [
      {
        "box_2d": "[ymin, xmin, ymax, xmax]",
        "label": "<label for the object>",
        "mask": "data:image/png;base64,<base64 encoded PNG mask>"
      },
      ...
    ]

    The box_2d coordinates should be normalized to 0-1000 and must be integers.
    The mask should be a base64 encoded PNG image where non-zero pixels indicate
    the mask.""" % segmentation_texts)

    start_time = time.time()
    settings=types.GenerateContentConfig(temperature=0.5)
    print("Raw Model Response Text:")

    try:
        json_output = call_gemini_robotics_er([image], prompt, settings)

    except Exception as e:
        print(f"Gemini Error: {e}")
    
    else:
        print(f"\nTotal processing time: {(time.time() - start_time):.4f} seconds")
    
        try:
            mask_tensor = parse_masks_to_tensor(
                json_output, height, width
            )
            print(f"Success! Tensor Shape: {mask_tensor.shape}")

            if mask_tensor.shape[0] > 0:
                _, axes = plt.subplots(1, mask_tensor.shape[0] + 1, figsize=(15, 10))

                axes[0].imshow(image)
                axes[0].set_title("Original Image")
                axes[0].axis('off')

                for i in range(mask_tensor.shape[0]):
                    axes[i+1].imshow(mask_tensor[i], cmap='gray')
                    axes[i+1].set_title(f"Mask {i+1}")
                    axes[i+1].axis('off')

                plt.show()
            else:
                print("No masks found to plot.")

        except Exception as e:
            print(f"An error occurred: {e}")

    try:
        segmentation_masks = parse_segmentation_masks(
            json_output, height, width
        )
        print(f"Successfully parsed {len(segmentation_masks)} segmentation masks.")

        #annotated_img = plot_segmentation_masks(
        #    image.convert("RGBA"), segmentation_masks
        #)
        #display.display(annotated_img)

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
    except Exception as e:
        print(f"An error occurred during mask processing or plotting: {e}")
        masks = mask_tensor
    return masks, phrases

def get_gemini_output_wrist(image, segmentation_texts):
    phrases = segmentation_texts
    segmentation_texts = ", ".join(segmentation_texts) #queries
    width, height = image.size

    prompt = textwrap.dedent("""\
    Provide the segmentation masks for the following objects seen from above in this image: %s.

    The answer should follow the JSON format:
    [
      {
        "box_2d": "[ymin, xmin, ymax, xmax]",
        "label": "<label for the object>",
        "mask": "data:image/png;base64,<base64 encoded PNG mask>"
      },
      ...
    ]

    The box_2d coordinates should be normalized to 0-1000 and must be integers.
    The mask should be a base64 encoded PNG image where non-zero pixels indicate
    the mask.""" % segmentation_texts)

    start_time = time.time()
    settings=types.GenerateContentConfig(temperature=0.5)
    print("Raw Model Response Text:")

    try:
        json_output = call_gemini_robotics_er([image], prompt, settings)

    except Exception as e:
        print(f"Gemini Error: {e}")
    
    else:
        print(f"\nTotal processing time: {(time.time() - start_time):.4f} seconds")
    
        try:
            mask_tensor = parse_masks_to_tensor(
                json_output, height, width
            )
            print(f"Success! Tensor Shape: {mask_tensor.shape}")

            if mask_tensor.shape[0] > 0:
                _, axes = plt.subplots(1, mask_tensor.shape[0] + 1, figsize=(15, 10))

                axes[0].imshow(image)
                axes[0].set_title("Original Image")
                axes[0].axis('off')

                for i in range(mask_tensor.shape[0]):
                    axes[i+1].imshow(mask_tensor[i], cmap='gray')
                    axes[i+1].set_title(f"Mask {i+1}")
                    axes[i+1].axis('off')

                plt.show()
            else:
                print("No masks found to plot.")

        except Exception as e:
            print(f"An error occurred: {e}")

    try:
        segmentation_masks = parse_segmentation_masks(
            json_output, height, width
        )
        print(f"Successfully parsed {len(segmentation_masks)} segmentation masks.")

        #annotated_img = plot_segmentation_masks(
        #    image.convert("RGBA"), segmentation_masks
        #)
        #display.display(annotated_img)

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
    except Exception as e:
        print(f"An error occurred during mask processing or plotting: {e}")
        masks = mask_tensor
    return masks, phrases
