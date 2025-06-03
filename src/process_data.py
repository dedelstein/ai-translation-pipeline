import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Custom paddle ocr is from: https://github.com/project-AIDA/Finnish_PaddleOCR

# Translation is from: https://huggingface.co/Helsinki-NLP/opus-mt-fi-en

# Lama inpainting is from: https://github.com/enesmsahin/simple-lama-inpainting

def translate_text(text, translator_tokenizer, translator_model, decoding_params={}):
    inputs = translator_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated_tokens = translator_model.generate(**inputs,  **decoding_params )
    translated_texts = [translator_tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]
    return translated_texts

def extract_regions_from_image(image_input, custom_ocr_system, polygon_orderer):
    if isinstance(image_input, str):
        print(f"\nProcessing image: {image_input}")
        image_data = cv2.imread(image_input)
        # It's good practice to convert to RGB as PaddleOCR works well with it.
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, np.ndarray):
        # If input is already a numpy array, use it directly.
        # We assume it's already in RGB format from app.py
        image_data = image_input

    original_bgr_image = image_data

    print("Performing OCR with custom PaddleOCR...")

    extracted_text_regions = ocr_with_custom_paddle(
        original_bgr_image,
        custom_ocr_system,
        polygon_orderer,
        reorder_texts=True
    )
    if not extracted_text_regions:
        print("No text found by PaddleOCR.")

    return extracted_text_regions

def ocr_with_custom_paddle(image_data_bgr: np.ndarray,
                           ocr_engine,
                           order_module,
                           use_angle_cls: bool = True,
                           reorder_texts: bool = True) -> list:
    """
    Implementation cribbed from:
        https://github.com/project-AIDA/Finnish_PaddleOCR
    """

    if ocr_engine is None:
        print("Custom PaddleOCR engine not initialized.")
        return []
    if image_data_bgr is None or image_data_bgr.size == 0:
        print("Empty image data passed to PaddleOCR.")
        return []

    try:
        ocr_result = ocr_engine.ocr(image_data_bgr, cls=use_angle_cls)

        if ocr_result is None or not ocr_result or ocr_result[0] is None:
             return []

        processed_result = ocr_result[0]

        if reorder_texts and order_module and processed_result:
            # The OrderPolygons class from api_onnx.py expects boxes in a specific format:
            paddle_boxes = [item[0] for item in processed_result] # item[0] is the list of points for the polygon

            # Convert PaddleOCR polygons to simple bounding rectangles [x_max, x_min, y_max, y_min]
            simple_boxes_for_ordering = []
            for poly_points in paddle_boxes:
                np_points = np.array(poly_points, dtype=np.float32)
                x, y, w, h = cv2.boundingRect(np_points)
                x_min, y_min = x, y
                x_max, y_max = x + w, y + h
                simple_boxes_for_ordering.append([x_max, x_min, y_max, y_min])

            if simple_boxes_for_ordering:
                new_order_indices = order_module.order(simple_boxes_for_ordering) 
                processed_result = [processed_result[i] for i in new_order_indices]
            else:
                 pass
        
        return processed_result # This will be like [[box_coords, (text, conf)], ...]
    except Exception as e:
        print(f"Error during custom PaddleOCR inference: {e}")
        return []

def inpaint_with_lama(img, regions, simple_lama):

    inpainting_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(inpainting_mask, regions, 255)

    dilation_kernel_size = 3
    dilation_iterations = 1
    kernel_dilate = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    inpainting_mask_dilated = cv2.dilate(inpainting_mask, kernel_dilate, iterations=dilation_iterations)

    inpainted = simple_lama(img, inpainting_mask_dilated)

    return inpainted

# Claude sonnet helped me w the render_text_to_img fns.  Sorry i have finals this week!!

def get_background_brightness(img: Image.Image, box: np.ndarray) -> float:
    """Calculate average brightness of region (0-255)."""
    x0, y0 = box.min(axis=0)
    x1, y1 = box.max(axis=0)
    region = np.array(img.crop((x0, y0, x1, y1)))
    
    # If image is RGB, convert to grayscale using standard coefficients
    if len(region.shape) == 3:
        return np.mean(region[:, :, 0] * 0.299 + 
                      region[:, :, 1] * 0.587 + 
                      region[:, :, 2] * 0.114)
    return np.mean(region)

def estimate_font_size(text: str, box_w: int, box_h: int, font_path: str, 
                      min_size: int = 10) -> int:
    area = box_w * box_h
    char_count = len(text) or 1
    target_char_area = (area * 0.65) / char_count
    estimated_size = int(np.sqrt(target_char_area))
    return max(min_size, estimated_size)

def render_text_in_regions(img: Image.Image, text: list, regions: np.ndarray,
                         font_path: str = "arial.ttf") -> Image.Image:
    """Render each text element into its corresponding region with maximum fitting font size.
    Returns a new image with the rendered text."""
    
    # Create a copy of the input image
    result = img.copy()
    draw = ImageDraw.Draw(result)
    
    for t, box in zip(text, regions):
        x0, y0 = box.min(axis=0)
        x1, y1 = box.max(axis=0)
        w, h = x1 - x0, y1 - y0
        
        # Quick brightness check
        region = np.array(img.crop((x0, y0, x1, y1)))
        brightness = np.mean(region) if len(region.shape) == 2 else \
                    np.mean(region[:, :, 0] * 0.299 + 
                           region[:, :, 1] * 0.587 + 
                           region[:, :, 2] * 0.114)
        text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)
        
        # Direct font size estimation
        font_size = estimate_font_size(t, w, h, font_path)
        font = ImageFont.truetype(font_path, font_size)
        
        # Center text
        bbox = draw.textbbox((0, 0), t, font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = x0 + (w - text_w) // 2
        y = y0 + (h - text_h) // 2
        
        draw.text((x, y), t, font=font, fill=text_color)
    
    return result