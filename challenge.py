"""
1. Load each image from /data/
2. Extract all visible Finnish text
3. Translate to English
4. Draw the translated text back onto the image
5. Save the result to /output/
"""
import argparse
import cv2
import glob
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from datasets import Dataset

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from paddleocr import PaddleOCR
from simple_lama_inpainting import SimpleLama

from models.order import OrderPolygons
from src.clean_data import filter_results, clean_translations
from src.process_data import extract_regions_from_image, render_text_in_regions, inpaint_with_lama, translate_text
from src.score import score_ocr, score_translation

ONNX_MODEL_BASE_PATH = './models/aida_onnx'
DB18modelPath = "./models/DB_TD500_resnet18.onnx"

img_dir = "./data"
output_dir = "./output"

deepl_json_file_path = "./expected/deepl_translations.json"
finnish_json_file_path = "./expected/evaluation_data.json"

MAX_LEN_MULTIPLIER = 3 # if translated is 3x longer than Finnish
MIN_FINNISH_LEN_FOR_CHECK = 3

def make_dataset(img_dir=img_dir):
    files = sorted(glob.glob(os.path.join(img_dir, "*")))
    return Dataset.from_dict({"file_name": files})

def extract_text(image_path):
    extracted_text_regions = extract_regions_from_image(image_path)
    return [region[1][0] for region in extracted_text_regions]

def render_translated_text(image_path, translated_blocks):
    image = Image.open(image_path).convert("RGB")
    text, regions = translated_blocks
    return render_text_in_regions(image, text, regions)

def main():

    parser = argparse.ArgumentParser(description="Translate Finnish text in images to English.")
    parser.add_argument(
        '--edit',
        action='store_true',
        help="Enable interactive editing of translations before rendering."
    )
    args = parser.parse_args()

    ds = make_dataset()

    with open(deepl_json_file_path, 'r', encoding='utf-8') as f:
        deepl_translations = json.load(f)

    with open(finnish_json_file_path, 'r', encoding='utf-8') as f:
        ocr_ground_truths = json.load(f)

    onnx_rec_model_path = os.path.join(ONNX_MODEL_BASE_PATH, 'rec_onnx', 'model.onnx')
    onnx_det_model_path = os.path.join(ONNX_MODEL_BASE_PATH, 'det_onnx', 'model.onnx')
    onnx_cls_model_path = os.path.join(ONNX_MODEL_BASE_PATH, 'cls_onnx', 'model.onnx')

    custom_ocr_system = PaddleOCR(
            lang='latin',       # As per repo
            show_log=False,
            det=True,
            use_angle_cls=True,
            rec_model_dir=onnx_rec_model_path,
            det_model_dir=onnx_det_model_path,
            cls_model_dir=onnx_cls_model_path,
            use_gpu=False,
            use_onnx=True,
            det_db_unclip_ratio=2.0,
            det_db_box_thresh=.5,
            det_db_thresh=.3,
            det_db_use_dilation=True, 
            )
    print("Custom PaddleOCR system initialized successfully.")

    polygon_orderer = OrderPolygons()

    translator_tokenizer = AutoTokenizer.from_pretrained("./models/tokenizer")
    translator_model = AutoModelForSeq2SeqLM.from_pretrained("./models/hknlp")

    simple_lama = SimpleLama()

    ocr_scores = []
    bleu_scores = []

    for item in ds.iter(1):
        img_path = item['file_name'][0]
        orig_img = Image.open(img_path).convert("RGB")
        raw_results = extract_regions_from_image(img_path, custom_ocr_system, polygon_orderer)

        # Filter the results
        filtered_results, finnish_text_segments = filter_results(raw_results)
        regions = np.asarray([r[0] for r in filtered_results], dtype=np.int32)
        finnish_text = finnish_text_segments

        inpainting_mask = np.zeros(np.array(orig_img).shape[:2], dtype=np.uint8)
        polygon_points = regions
        cv2.fillPoly(inpainting_mask, polygon_points, 255)
        plt.imshow(inpainting_mask, cmap='gray')

        inpainted = inpaint_with_lama(np.array(orig_img), regions, simple_lama)

        ocr_score = score_ocr(img_path, finnish_text, ocr_ground_truths)
        translated_text = translate_text(finnish_text, translator_tokenizer, translator_model)
        translated_text = clean_translations(finnish_text, translated_text, translator_tokenizer, translator_model)
        bleu_score =score_translation(img_path, translated_text, deepl_translations)

        if args.edit and finnish_text: # Check if edit mode is on and there's text
            final_translated_texts = []
            print(f"\n--- Editing translations for: {os.path.basename(img_path)} ---")
            for i, (fi_text, en_text) in enumerate(zip(finnish_text, translated_text)):
                print(f"\nSegment {i+1}/{len(finnish_text)}:")
                print(f"  Finnish (OCR): \"{fi_text}\"")
                print(f"  Model Output : \"{en_text}\"")
                user_input = input("  Enter corrected English (or press Enter to keep model output): ")
                if user_input.strip():
                    final_translated_texts.append(user_input.strip())
                else:
                    final_translated_texts.append(en_text)
            print("--- Finished editing for this image ---")
        else:
            final_translated_texts = translated_text

        rendered = render_text_in_regions(inpainted, final_translated_texts, regions)
        ocr_scores.append(ocr_score)
        bleu_scores.append(bleu_score)
        rendered.save(os.path.join(output_dir,"translated_") + img_path.replace("./data/", ""))

    print(ocr_scores)
    print(bleu_scores)


if __name__ == "__main__":
    main()
