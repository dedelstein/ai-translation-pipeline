# Finnish-to-English Image Translation Pipeline

I built this over a weekend as a complete on-premise image translation system that processes Finnish text in images and renders English translations back onto the original image. No cloud APIs, no external dependencies—everything runs locally.

## What Does This Thing Actually Do?

The system processes images containing Finnish text through a five-stage pipeline:
1. Detects all text regions in the image
2. Performs OCR to extract the Finnish text
3. Intelligently removes the original text from the image using inpainting
4. Translates the Finnish to English using a local transformer model
5. Renders the English translation back onto the processed image

The entire pipeline runs locally without external API dependencies. It's essentially a self-contained version of camera-based translation tools, optimized for Finnish-English translation.

## Why Build This?

I wanted to explore the challenges of integrating multiple heavy ML models into a functional end-to-end system. Building a completely self-sufficient translation pipeline also eliminates the dependency on external services and their associated costs and latency.

This project provided hands-on experience with:
- **Model integration challenges** — Orchestrating PaddleOCR, LaMa, and Helsinki-NLP models in sequence
- **Resource management** — Running multiple deep learning models efficiently on CPU
- **Pipeline orchestration** — Ensuring clean data flow between processing stages

## The Tech Stack

Here's what's under the hood:

| What It Does | How It Does It |
|--------------|----------------|
| **Text Detection** | PaddleOCR (ONNX) — finds text boxes in the image |
| **Text Recognition** | PaddleOCR (ONNX) — actually reads the Finnish text |
| **Text Removal** | LaMa Inpainting — magically erases text and fills in the background |
| **Translation** | Helsinki-NLP Transformer — Finnish → English translation |
| **Text Rendering** | Pillow/OpenCV — draws the English text back onto the image |

## The CPU vs GPU Trade-off

This implementation uses CPU inference, which trades speed for deployment flexibility and resource independence. While GPU acceleration or cloud-based inference would offer better performance, the CPU approach demonstrates the feasibility of running complex AI pipelines in resource-constrained or offline environments.

The design prioritizes self-sufficiency and universal deployment capability over raw processing speed.

## Project Layout

```
.
├── data/                  # Drop your Finnish images here
├── output/                # Translated images end up here
├── models/                # All the ML models live here
│   ├── aida_onnx/        # PaddleOCR detection models
│   ├── hknlp/            # Helsinki-NLP translation model
│   └── tokenizer/        # Tokenizer for the translation model
├── expected/              # Reference data for evaluation
└── challenge.py           # The main script that does all the magic
```

## Getting Started

**1. Grab the code:**
```bash
git clone <your-repo-url>
cd <repo-name>
```

**2. Install the dependencies:**
```bash
pip install -r requirements.txt
```
*(You'll need to create the requirements.txt from the imports in challenge.py)*

**3. Download the models:**
Make sure you've got all the pre-trained models in the right places. Check the project structure above—the models directory needs to be populated with the PaddleOCR ONNX models, Helsinki-NLP transformer, etc.

## Running It

**Basic usage** (processes all images in ./data):
```bash
python challenge.py
```

**Interactive mode** (allows manual review/editing of translations):
```bash
python challenge.py --edit
```

In edit mode, you'll get prompted for each text segment:
```
--- Editing translations for: some_finnish_sign.jpg ---

Segment 1/3:
  Finnish (OCR): "VARO HEIKKOA JÄÄTÄ"
  Model Output : "BEWARE OF WEAK ICE"
  Enter corrected English (or press Enter to keep): WARNING: THIN ICE

--- Moving to next segment ---
```

## Implementation Notes

Building this system highlighted the complexity of multi-model integration beyond individual model performance. The text detection and inpainting work reliably, and the translation quality is solid for a local transformer model.

The most challenging aspect was accurate text positioning during the rendering stage. Proper font matching, text wrapping, and handling various text orientations required careful geometric calculations and visual calibration.

## Potential Extensions

- Support for additional language pairs
- GPU acceleration for improved throughput  
- Enhanced font matching algorithms
- Web interface for easier interaction

The modular design makes it straightforward to swap components or extend functionality for different use cases.