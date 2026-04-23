import os
import threading
from pathlib import Path
import pytesseract
from PIL import Image, ImageGrab, ImageEnhance
import customtkinter as ctk
from tkinter import filedialog, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

_trocr_processor = None
_trocr_model = None
_spell = None
_progress = {"current": 0, "total": 0, "status": "", "result": None, "error": None, "done": False}


# ── OCR backend ────────────────────────────────────────────────────────────────

def get_trocr():
    global _trocr_processor, _trocr_model
    if _trocr_model is None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        _progress["status"] = "Loading TrOCR model — first time is a ~1.5GB download..."
        _trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
        _trocr_model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-large-handwritten",
            low_cpu_mem_usage=False
        )
    return _trocr_processor, _trocr_model

def preprocess_image(image):
    if image.width < 1200:
        scale = 1200 / image.width
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)
    return ImageEnhance.Contrast(image).enhance(1.5)

def detect_line_crops(image):
    """Use Tesseract's layout analysis to find line bounding boxes, then crop each line."""
    data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT,
        config="--psm 6 --oem 3"
    )

    # Group word boxes by their line key
    line_boxes = {}
    for i in range(len(data["text"])):
        if data["conf"][i] == -1:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        if key not in line_boxes:
            line_boxes[key] = [x, y, x + w, y + h]
        else:
            line_boxes[key][0] = min(line_boxes[key][0], x)
            line_boxes[key][1] = min(line_boxes[key][1], y)
            line_boxes[key][2] = max(line_boxes[key][2], x + w)
            line_boxes[key][3] = max(line_boxes[key][3], y + h)

    if not line_boxes:
        return [image]

    pad = 10
    crops = []
    for box in sorted(line_boxes.values(), key=lambda b: b[1]):
        x1 = max(0, box[0] - pad)
        y1 = max(0, box[1] - pad)
        x2 = min(image.width,  box[2] + pad)
        y2 = min(image.height, box[3] + pad)
        if x2 - x1 > 10 and y2 - y1 > 10:
            crops.append(image.crop((x1, y1, x2, y2)))

    return crops if crops else [image]

def trocr_extract(image):
    import torch
    processor, model = get_trocr()
    image = preprocess_image(image)
    _progress["status"] = "Detecting text regions..."
    lines = detect_line_crops(image)
    _progress["total"] = len(lines)

    results = []
    batch_size = 4
    for batch_start in range(0, len(lines), batch_size):
        batch = lines[batch_start:batch_start + batch_size]
        _progress["current"] = batch_start + len(batch)
        _progress["status"] = f"Processing lines {batch_start + 1}–{batch_start + len(batch)} of {len(lines)}..."
        pixel_values = processor(
            images=[img.convert("RGB") for img in batch],
            return_tensors="pt"
        ).pixel_values
        with torch.inference_mode():
            generated_ids = model.generate(pixel_values, num_beams=2, max_new_tokens=64)
        texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        results.extend(t for t in texts if t.strip())

    return "\n".join(results)

def spell_correct(text):
    global _spell
    if _spell is None:
        from spellchecker import SpellChecker
        _spell = SpellChecker()
    corrected_lines = []
    for line in text.split("\n"):
        corrected_words = []
        for word in line.split():
            prefix, core, suffix = "", word, ""
            while core and not core[0].isalpha():
                prefix += core[0]; core = core[1:]
            while core and not core[-1].isalpha():
                suffix = core[-1] + suffix; core = core[:-1]
            if core:
                fix = _spell.correction(core.lower())
                if fix and fix != core.lower():
                    fix = fix.capitalize() if core[0].isupper() else fix
                    corrected_words.append(prefix + fix + suffix)
                    continue
            corrected_words.append(word)
        corrected_lines.append(" ".join(corrected_words))
    return "\n".join(corrected_lines)

def run_ocr(image_source, is_path, mode):
    image = Image.open(image_source).convert("RGB") if is_path else image_source.convert("RGB")
    if mode == "printed":
        _progress["status"] = "Running OCR..."
        return pytesseract.image_to_string(image)
    text = trocr_extract(image)
    _progress["status"] = "Spell checking..."
    return spell_correct(text)


# ── GUI ────────────────────────────────────────────────────────────────────────

class App(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self):
        super().__init__()
        try:
            self.TkdndVersion = TkinterDnD._tkdnd_lib.TkdndVersion
        except AttributeError:
            self.TkdndVersion = TkinterDnD._require(self)

        self.title("OCR — Image to Text")
        self.geometry("740x700")
        self.minsize(600, 520)
        self._mode = "printed"

        self._build_ui()
        self.bind("<Control-v>", self.paste_image)

    def _build_ui(self):
        # ── Header ──
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=24, pady=(22, 10))

        ctk.CTkLabel(
            header, text="OCR — Image to Text",
            font=ctk.CTkFont(size=22, weight="bold")
        ).pack(side="left")

        self.mode_btn = ctk.CTkSegmentedButton(
            header,
            values=["Printed", "Handwritten"],
            command=self._on_mode_change,
            width=210,
            font=ctk.CTkFont(size=13)
        )
        self.mode_btn.set("Printed")
        self.mode_btn.pack(side="right")

        # ── Drop zone ──
        self.drop_zone = ctk.CTkFrame(
            self, corner_radius=14,
            border_width=2, border_color=("gray75", "gray35"),
            fg_color=("gray95", "gray17"), height=130
        )
        self.drop_zone.pack(fill="x", padx=24, pady=(0, 10))
        self.drop_zone.pack_propagate(False)

        self.drop_label = ctk.CTkLabel(
            self.drop_zone,
            text="Drop an image here",
            font=ctk.CTkFont(size=15),
            text_color=("gray55", "gray55")
        )
        self.drop_label.place(relx=0.5, rely=0.5, anchor="center")

        for w in (self.drop_zone, self.drop_label):
            w.drop_target_register(DND_FILES)
            w.dnd_bind("<<Drop>>", self._on_drop)

        # ── Buttons ──
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(pady=6)

        self.btn_browse = ctk.CTkButton(btn_row, text="Browse", width=130, height=36, command=self.browse_file)
        self.btn_paste  = ctk.CTkButton(btn_row, text="Paste  Ctrl+V", width=150, height=36, command=self.paste_image)
        self.btn_copy   = ctk.CTkButton(
            btn_row, text="Copy Text", width=130, height=36,
            fg_color=("gray70", "gray30"), hover_color=("gray60", "gray40"),
            command=self.copy_text
        )
        for btn in (self.btn_browse, self.btn_paste, self.btn_copy):
            btn.pack(side="left", padx=5)
        self._all_btns = [self.btn_browse, self.btn_paste, self.btn_copy]

        # ── Progress bar (hidden until active) ──
        self.progress_bar = ctk.CTkProgressBar(self, height=6, corner_radius=3)

        # ── Output ──
        ctk.CTkLabel(
            self, text="Extracted Text",
            font=ctk.CTkFont(size=13, weight="bold"), anchor="w"
        ).pack(fill="x", padx=26, pady=(10, 3))

        self.output_box = ctk.CTkTextbox(
            self, font=ctk.CTkFont(family="Segoe UI", size=12),
            wrap="word", corner_radius=10
        )
        self.output_box.pack(fill="both", expand=True, padx=24, pady=(0, 8))

        # ── Status bar ──
        self.status_label = ctk.CTkLabel(
            self, text="Drop an image, paste (Ctrl+V), or click Browse.",
            font=ctk.CTkFont(size=11),
            text_color=("gray50", "gray55"), anchor="w"
        )
        self.status_label.pack(fill="x", padx=26, pady=(0, 16))

    # ── Event handlers ──────────────────────────────────────────────────────────

    def _on_mode_change(self, value):
        self._mode = "printed" if value == "Printed" else "handwritten"

    def _on_drop(self, event):
        path = event.data.strip().strip("{}")
        if not os.path.exists(path):
            self._set_status("File not found.", "red")
            return
        if Path(path).suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif"}:
            self._set_status("Unsupported file type. Please drop an image file.", "red")
            return
        self.drop_label.configure(text=Path(path).name)
        self._launch(path, is_path=True, stem=Path(path).stem)

    def browse_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif"), ("All files", "*.*")]
        )
        if not path:
            return
        if Path(path).suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif"}:
            self._set_status("Unsupported file type. Please select an image file.", "red")
            return
        self.drop_label.configure(text=Path(path).name)
        self._launch(path, is_path=True, stem=Path(path).stem)

    def paste_image(self, _event=None):
        image = ImageGrab.grabclipboard()
        if image is None:
            self._set_status("No image found in clipboard.", "red")
            return
        self.drop_label.configure(text="(from clipboard)")
        self._launch(image, is_path=False, stem="clipboard")

    def copy_text(self):
        text = self.output_box.get("1.0", "end").strip()
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)
            self._set_status("Copied to clipboard.")

    # ── Processing ──────────────────────────────────────────────────────────────

    def _launch(self, source, is_path, stem):
        mode = self._mode
        _progress.update({"current": 0, "total": 0, "status": "Starting...", "result": None, "error": None, "done": False})
        for btn in self._all_btns:
            btn.configure(state="disabled")
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.pack(fill="x", padx=24, pady=(0, 4))
        self.progress_bar.start()
        self._set_status("Starting...")

        def worker():
            try:
                text = run_ocr(source, is_path=is_path, mode=mode)
                _progress["result"] = (text, stem)
            except Exception as e:
                _progress["error"] = e
            finally:
                _progress["done"] = True

        threading.Thread(target=worker, daemon=True).start()
        self.after(100, self._poll)

    def _poll(self):
        if _progress["status"]:
            self._set_status(_progress["status"])

        if _progress["total"] > 0:
            self.progress_bar.stop()
            self.progress_bar.configure(mode="determinate")
            self.progress_bar.set(_progress["current"] / _progress["total"])

        if _progress["done"]:
            self.progress_bar.stop()
            self.progress_bar.pack_forget()
            for btn in self._all_btns:
                btn.configure(state="normal")
            if _progress["error"]:
                messagebox.showerror("Error", str(_progress["error"]))
                self._set_status("Failed to process image.", "red")
            else:
                text, stem = _progress["result"]
                self._save_and_display(text, stem)
            return

        self.after(100, self._poll)

    def _save_and_display(self, text, stem):
        self.output_box.delete("1.0", "end")
        self.output_box.insert("1.0", text)
        desktop = Path.home() / "Desktop"
        output_file = desktop / f"{stem}_ocr_output.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)
        self._set_status(f"Saved to: {output_file}", ("green", "lightgreen"))

    def _set_status(self, text, color=None):
        self.status_label.configure(
            text=text,
            text_color=color or ("gray50", "gray55")
        )


if __name__ == "__main__":
    # Preload TrOCR in the background so it's ready when the user drops an image
    threading.Thread(target=get_trocr, daemon=True).start()
    App().mainloop()
