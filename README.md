# Offline OCR

A desktop app for extracting text from images — fully offline. Supports both **printed** and **handwritten** text.

## Download

**[⬇ Download Offline_OCR.zip](https://github.com/karljuderojas/Offline_OCR/raw/main/Offline_OCR.zip)**

Extract the zip and follow the [Installation](#installation) steps below.

## Features

- **Printed text** — fast extraction via [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- **Handwritten text** — high-accuracy extraction via Microsoft's [TrOCR](https://huggingface.co/microsoft/trocr-large-handwritten) model
- Drag-and-drop images onto the window
- Paste directly from clipboard (`Ctrl+V`)
- Browse for image files
- Automatic spell correction on handwritten output
- Results auto-saved as a `.txt` file to your Desktop
- Copy output to clipboard in one click

## Requirements

- Python 3.9+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed at `C:\Program Files\Tesseract-OCR\tesseract.exe`

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/karljuderojas/Offline_OCR.git
   cd Offline_OCR
   ```

2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install Tesseract OCR for Windows from [here](https://github.com/UB-Mannheim/tesseract/wiki). Use the default install path (`C:\Program Files\Tesseract-OCR\`).

## Usage

**Option A — double-click launcher:**
```
run.bat
```

**Option B — run directly:**
```
python ocr_script.py
```

### Steps

1. Select a mode: **Printed** or **Handwritten** (top-right toggle)
2. Load an image using one of:
   - **Drop** an image file onto the drop zone
   - **Browse** to select a file
   - **Paste** an image from clipboard (`Ctrl+V`)
3. Wait for processing to complete — a progress bar shows status
4. The extracted text appears in the output box
5. The result is automatically saved to your Desktop as `<filename>_ocr_output.txt`
6. Click **Copy Text** to copy the output to clipboard

### Notes

- The first time you use **Handwritten** mode, TrOCR (~1.5 GB) will be downloaded automatically and cached locally. Subsequent runs are instant.
- Supported image formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.gif`

## Dependencies

| Package | Purpose |
|---|---|
| `pytesseract` | Printed OCR via Tesseract |
| `transformers` + `torch` | Handwritten OCR via TrOCR |
| `Pillow` | Image loading and preprocessing |
| `customtkinter` | GUI framework |
| `tkinterdnd2` | Drag-and-drop support |
| `pyspellchecker` | Spell correction for handwritten output |
