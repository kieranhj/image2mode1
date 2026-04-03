#!/usr/bin/env python3
"""
gradio_ui.py — Web UI for palsearch.py

Provides live preprocessing preview, full BBC Mode 1 conversion,
and .bin download for every option exposed by the CLI.

Run with:
    python palsearch/gradio_ui.py

Then open http://localhost:7860 in your browser.

Requires:
    pip install gradio Pillow numpy
"""

import base64
import io
import os
import sys
import tempfile
import traceback
import zipfile

import numpy as np

try:
    import gradio as gr
except ImportError:
    print("ERROR: gradio not installed.  Run: pip install gradio", file=sys.stderr)
    sys.exit(1)

from PIL import Image, ImageEnhance, ImageOps, ImageFilter

# Ensure palsearch module is importable from the same directory
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from palsearch import (
    process_image, _fit_image,
    SCREEN_W, SCREEN_H, CHANGE_PER_ROW, _AUTO_DITHER_THRESHOLD,
)


# ── Preprocessing pipeline (mirrors process_image — used for live preview) ────

def _apply_preprocessing(img_np, resize,
                          autolevel, input_gamma, contrast, brightness,
                          saturation, hue, denoise, posterise, sharpen):
    """
    Apply the full preprocessing pipeline to a numpy image array.
    Returns a 320×256 PIL Image, or None if img_np is None.
    Fast enough for live UI preview (< 200 ms per call).
    """
    if img_np is None:
        return None
    img = Image.fromarray(img_np.astype(np.uint8)).convert('RGB')
    if img.size != (SCREEN_W, SCREEN_H):
        img = _fit_image(img, resize)

    # Prep 2 — Auto-levels
    if autolevel:
        img = ImageOps.autocontrast(img)

    # Prep 5 — Input gamma (gamma curve on pixels, separate from internal scoring gamma)
    if input_gamma != 1.0:
        exp = 1.0 / max(input_gamma, 0.01)
        lut = [int(min(255, max(0, (i / 255.0) ** exp * 255.0 + 0.5)))
               for i in range(256)]
        img = img.point(lut * 3)

    # Prep 3 — Contrast
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)

    # Prep 4 — Brightness
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)

    # Prep 1 — Saturation (most impactful; after levels/contrast/brightness)
    if saturation != 1.0:
        img = ImageEnhance.Color(img).enhance(saturation)

    # Prep 6 — Hue rotation via vectorised numpy RGB→HSV→RGB
    if hue != 0.0:
        h_shift = (hue % 360.0) / 360.0
        f = np.array(img, dtype=np.float32) / 255.0
        r, g, b = f[:, :, 0], f[:, :, 1], f[:, :, 2]
        cmax  = np.maximum(np.maximum(r, g), b)
        cmin  = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin
        hh    = np.zeros_like(r)
        mask  = delta > 0
        mr = mask & (cmax == r)
        mg = mask & (cmax == g)
        mb = mask & (cmax == b)
        hh[mr] = ((g[mr] - b[mr]) / delta[mr]) % 6.0
        hh[mg] = ((b[mg] - r[mg]) / delta[mg]) + 2.0
        hh[mb] = ((r[mb] - g[mb]) / delta[mb]) + 4.0
        hh  = (hh / 6.0 + h_shift) % 1.0
        ss  = np.where(cmax > 0, delta / np.where(cmax > 0, cmax, 1.0), 0.0)
        vv  = cmax
        hh6 = hh * 6.0
        hi  = np.floor(hh6).astype(int) % 6
        ff  = hh6 - np.floor(hh6)
        p   = vv * (1.0 - ss)
        q   = vv * (1.0 - ss * ff)
        t   = vv * (1.0 - ss * (1.0 - ff))
        out = np.zeros_like(f)
        for ch, (rv, gv, bv) in enumerate([
                (vv, t, p), (q, vv, p), (p, vv, t),
                (p, q, vv), (t, p, vv), (vv, p, q)]):
            m = hi == ch
            out[:, :, 0][m] = rv[m]
            out[:, :, 1][m] = gv[m]
            out[:, :, 2][m] = bv[m]
        img = Image.fromarray((out * 255.0).clip(0, 255).astype(np.uint8), 'RGB')

    # Prep 7 — Denoise
    if denoise:
        img = img.filter(ImageFilter.MedianFilter(size=3))

    # Prep 8 — Posterise (quantise to N bits then rescale to full 0-255)
    if posterise > 0:
        bits = int(posterise)
        levels = (1 << bits) - 1
        lut = [int(((v >> (8 - bits)) * 255 / levels) + 0.5)
               for v in range(256)]
        img = img.point(lut * 3)

    # Option 8 — Sharpen (last, so sharpening is on the final pixel values)
    if sharpen > 0:
        img = img.filter(ImageFilter.UnsharpMask(
            radius=1.5, percent=int(sharpen * 150), threshold=3))

    return img


# ── Full conversion (runs process_image, returns preview + .bin path) ─────────

def _convert(img_np,
             # Preprocessing
             resize, autolevel, input_gamma, contrast, brightness,
             saturation, hue, denoise, posterise, sharpen,
             # Dithering
             dither, randomness, mixno, chunk_size, auto_threshold,
             # Solver
             solver, changes, look_ahead, restarts, beam, anneal, smooth,
             progress=gr.Progress(track_tqdm=False)):

    if img_np is None:
        return None, None, "No image loaded."

    chunk_size = int(chunk_size)


    # Save input image to a temp PNG
    input_tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    input_tmp.close()
    Image.fromarray(img_np.astype(np.uint8)).convert('RGB').save(input_tmp.name)

    bin_path     = tempfile.NamedTemporaryFile(suffix='.bin',  delete=False).name
    preview_path = tempfile.NamedTemporaryFile(suffix='.png',  delete=False).name

    try:
        progress(0, desc="Running palette solver…")
        process_image(
            input_tmp.name, bin_path,
            dither            = dither,
            verbose           = False,
            preview_path      = preview_path,
            solver            = solver,
            resize            = resize,
            randomness        = int(randomness),
            look_ahead        = look_ahead,
            chunk_size        = chunk_size,
            changes_per_row   = int(changes),
            restarts          = int(restarts),
            mixno             = int(mixno),
            sharpen           = float(sharpen),
            auto_threshold    = float(auto_threshold),
            beam              = int(beam),
            anneal            = int(anneal),
            smooth            = float(smooth),
            saturation        = float(saturation),
            autolevel         = autolevel,
            contrast          = float(contrast),
            brightness        = float(brightness),
            input_gamma       = float(input_gamma),
            hue               = float(hue),
            denoise           = denoise,
            posterise         = int(posterise),
            progress_callback = lambda v: progress(v, desc=f"Solving… {v*100:.0f}%"),
        )
        progress(1.0, desc="Done!")

        preview_img = Image.open(preview_path).copy()
        preview_img = preview_img.resize(
            (preview_img.width * 3, preview_img.height * 3),
            Image.NEAREST)
        bin_size    = os.path.getsize(bin_path)
        sections    = SCREEN_H // chunk_size
        msg = (f"Done.  {bin_size} bytes  "
               f"({256 + int(changes) * sections} palette + 20480 screen).")

        # Build SSD + emulator link if the template exists and chunk_size=2
        ssd_path = None
        emu_html = ""
        if chunk_size == 2 and os.path.exists(_SSD_TEMPLATE):
            try:
                ssd_path, emu_html, _ = _build_ssd(bin_path)
            except Exception:
                pass   # SSD build is best-effort

        return preview_img, bin_path, ssd_path, emu_html, msg

    except Exception as exc:
        return None, None, None, "", f"Error: {exc}\n\n{traceback.format_exc()}"

    finally:
        for p in (input_tmp.name, preview_path):
            try:
                os.unlink(p)
            except OSError:
                pass


# ── Build SSD and launch emulator ─────────────────────────────────────────────

_REPO_ROOT = os.path.dirname(_here)   # parent of palsearch/
_SSD_TEMPLATE = os.path.join(_REPO_ROOT, 'raster-fx.ssd')


def _patch_ssd(template_path, pal_data, pic_data):
    """Patch PAL and PIC file data in a pre-built DFS .ssd image.

    Parses the DFS catalog (sectors 0-1) to find the PIC and PAL entries,
    then overwrites their data at the correct sector offsets and updates
    the file lengths in the catalog.  No beebasm needed.
    """
    ssd = bytearray(open(template_path, 'rb').read())
    num_files = ssd[0x105] // 8

    for i in range(num_files):
        name = ssd[0x008 + i*8 : 0x008 + i*8 + 7].decode('ascii').rstrip()
        off = 0x108 + i * 8
        mixed     = ssd[off + 6]
        start_sec = ((mixed & 0x03) << 8) | ssd[off + 7]
        byte_off  = start_sec * 256

        if name == 'PAL':
            ssd[off + 4] = len(pal_data) & 0xFF
            ssd[off + 5] = (len(pal_data) >> 8) & 0xFF
            ssd[off + 6] = (mixed & 0xCF) | (((len(pal_data) >> 16) & 0x03) << 4)
            ssd[byte_off:byte_off + len(pal_data)] = pal_data
        elif name == 'PIC':
            ssd[byte_off:byte_off + len(pic_data)] = pic_data

    return bytes(ssd)


def _build_ssd(bin_file):
    """Patch PIC+PAL into the SSD template and return an emulator link."""
    bin_data = open(bin_file, 'rb').read()
    pal_data = bin_data[:len(bin_data) - 20480]
    pic_data = bin_data[len(bin_data) - 20480:]

    ssd_data = _patch_ssd(_SSD_TEMPLATE, pal_data, pic_data)

    # Save patched SSD for download
    ssd_tmp = tempfile.NamedTemporaryFile(suffix='.ssd', delete=False)
    ssd_tmp.write(ssd_data)
    ssd_tmp.close()

    # Build emulator URL with the SSD embedded as base64 in the hash fragment.
    # jsbeeb's data: handler expects a ZIP containing an .ssd file.
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('raster-fx.ssd', ssd_data)
    b64 = base64.b64encode(zip_buf.getvalue()).decode('ascii')
    emu_url = f"https://bbc.xania.org/#autoboot&model=master&disc1=data:{b64}"

    link_html = (
        f'<a href="{emu_url}" target="_blank" '
        f'style="display:inline-block;width:100%;padding:10px;'
        f'background:#2563eb;color:white;border-radius:8px;'
        f'text-decoration:none;font-weight:bold;text-align:center">'
        f'▶ Run on BBC Micro</a>')

    return ssd_tmp.name, link_html, f"Built SSD ({len(ssd_data)} bytes)."


# ── Gradio UI layout ───────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="palsearch - BBC Master Mode 1") as demo:

        gr.Markdown(
            "# palsearch - BBC Master Mode 1 converter\n"
            "Converts images to BBC Master Mode 1 screen data with per-scanline "
            "palette changes.  Adjust the **Preprocessing** tab to see a live "
            "preview of the 320×256 source.  Click **Convert** to run the full "
            "palette solver and download the `.bin`."
        )

        # ── Image previews ─────────────────────────────────────────────────────
        # Left column: source images side-by-side (smaller, natural height).
        # Right column: BBC output — larger, the main result.
        # buttons=["fullscreen","download"] is set explicitly on output images
        # because Gradio 6.x only shows overlay buttons when they are named.
        with gr.Row():
            with gr.Column(scale=5):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 1 · Input")
                        input_img = gr.Image(
                            label="Upload source image",
                            type="numpy", image_mode="RGB",
                            buttons=["fullscreen", "download"])
                    with gr.Column():
                        gr.Markdown("### 2 · Preprocessed (320×256, live)")
                        prep_img = gr.Image(
                            label="Updates on every Preprocessing control change",
                            type="pil",
                            buttons=["fullscreen", "download"])
            with gr.Column(scale=6):
                gr.Markdown("### 3 · Converted (BBC Mode 1)")
                conv_img = gr.Image(
                    label="Click Convert to generate",
                    type="pil",
                    buttons=["fullscreen", "download"])
                convert_btn = gr.Button("Convert", variant="primary")
                emu_link    = gr.HTML()
                status_box  = gr.Textbox(label="Status", interactive=False,
                                         show_label=False)
                with gr.Row():
                    dl_file = gr.File(label="Download .bin", scale=1)
                    dl_ssd  = gr.File(label="Download .ssd", scale=1)

        # ── Parameter tabs ─────────────────────────────────────────────────────
        with gr.Tabs():

            # ── Tab 1 — Image Preprocessing (fast, drives live preview) ───────
            with gr.Tab("Preprocessing"):
                gr.Markdown(
                    "All controls on this tab update the **Preprocessed** preview "
                    "immediately — no solver is run.  BBC colours sit at the "
                    "corners of the RGB cube, so pushing pixels toward those "
                    "corners reduces dither noise.")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Tone & colour")

                        saturation = gr.Slider(
                            0.0, 3.0, value=1.0, step=0.05,
                            label="Saturation",
                            info="Boost colour saturation (1.0 = none). "
                                 "Unsaturated pixels are far from BBC colour corners and "
                                 "produce noisy dithering — 1.3–1.8 suits most photos.")

                        autolevel = gr.Checkbox(
                            value=False, label="Auto-levels",
                            info="Stretch each channel's histogram to the full 0–255 range. "
                                 "Fixes washed-out or dark images before dithering.")

                        contrast = gr.Slider(
                            0.0, 3.0, value=1.0, step=0.05,
                            label="Contrast",
                            info="Contrast multiplier (1.0 = none, >1 increases). "
                                 "Pushes midtones toward black/white, "
                                 "reducing quantisation error. Apply after Auto-levels.")

                        brightness = gr.Slider(
                            0.1, 3.0, value=1.0, step=0.05,
                            label="Brightness",
                            info="Brightness multiplier (1.0 = none). "
                                 "Simple exposure correction for over/underexposed images.")

                        input_gamma = gr.Slider(
                            0.2, 3.0, value=1.0, step=0.05,
                            label="Input gamma",
                            info="Gamma applied to image pixels before scoring (1.0 = none). "
                                 ">1 brightens midtones, <1 darkens. Separate from the "
                                 "internal INV_GAMMA=1.7 scoring curve.")

                        hue = gr.Slider(
                            -180.0, 180.0, value=0.0, step=1.0,
                            label="Hue rotation (degrees)",
                            info="Rotate all hues by this many degrees. Useful when a "
                                 "dominant colour falls between BBC primaries "
                                 "(e.g. orange → shift toward red or yellow).")

                    with gr.Column():
                        gr.Markdown("#### Detail & style")

                        sharpen = gr.Slider(
                            0.0, 2.0, value=0.0, step=0.05,
                            label="Sharpen",
                            info="Unsharp mask strength (0 = off, 0.5 mild, 1.0 strong). "
                                 "Recovers perceived detail lost in low-resolution dithering. "
                                 "Applied last, after all other preprocessing.")

                        denoise = gr.Checkbox(
                            value=False, label="Denoise",
                            info="Apply a 3×3 median filter to suppress noise before dithering. "
                                 "Prevents sensor/compression noise from being encoded into "
                                 "the dither pattern.")

                        posterise = gr.Slider(
                            0, 7, value=0, step=1,
                            label="Posterise (bits per channel, 0 = off)",
                            info="Reduce each channel to N bits (1 = 2 levels, 7 = 128 levels). "
                                 "Creates a flat, graphic-art look and simplifies the palette "
                                 "solver. Mainly a stylistic option.")

                        gr.Markdown("#### Layout")

                        resize = gr.Radio(
                            choices=["fit", "crop-left", "crop-right",
                                     "crop-top", "crop-bottom"],
                            value="fit",
                            label="Resize mode",
                            info="How to fit the source image to 320×256. "
                                 "'fit' letterboxes or pillarboxes (black bars). "
                                 "'crop-*' fills the screen and discards the excess edge.")

            # ── Tab 2 — Dithering ──────────────────────────────────────────────
            with gr.Tab("Dithering"):
                gr.Markdown(
                    "Controls how pixel colours are mapped and dithered to the "
                    "BBC palette entries available each scanline.")

                with gr.Row():
                    with gr.Column():
                        dither = gr.Radio(
                            choices=["ordered", "fs", "bn", "auto"],
                            value="ordered",
                            label="Dither method",
                            info="ordered — Bayer 2×2 matrix; smooth gradients, fast, "
                                 "visible pattern on flat regions.  "
                                 "fs — Floyd-Steinberg error diffusion; more detail but can "
                                 "produce noise on edges.  "
                                 "bn — blue-noise ordered dither; smoother than Bayer, "
                                 "no visible grid pattern.  "
                                 "auto — picks per section based on variance (see threshold).")

                        randomness = gr.Slider(
                            0, 255, value=64, step=1,
                            label="Ordered dither randomness",
                            info="Per-pixel random bias for ordered dither (0–255). "
                                 "Higher values break up flat-colour regions and reduce "
                                 "the visible Bayer pattern. 0 = pure ordered matrix.")

                        mixno = gr.Slider(
                            0, 3, value=0, step=1,
                            label="Ordered dither mix set",
                            info="Selects which set of BBC colour-pair combinations the "
                                 "ordered dither uses. 0 = lowest contrast (smooth blends "
                                 "between adjacent colours). Higher sets add black/white "
                                 "pairs — useful for high-contrast highlights/shadows.")

                    with gr.Column():
                        chunk_size = gr.Radio(
                            choices=[1, 2], value=2,
                            label="Chunk size (scanlines per section)",
                            info="2 — one palette per 2 scanlines (default). "
                                 "Produces a smaller output file (21,888 bytes). "
                                 "1 — per-scanline palette; requires stable raster hardware "
                                 "with sufficient cycles. Must be 1 for Vertical dithering. "
                                 "Output: 23,040 bytes.")

                        auto_threshold = gr.Slider(
                            0, 5000, value=_AUTO_DITHER_THRESHOLD, step=50,
                            label="Auto-dither variance threshold",
                            info="When dither = auto: sections whose per-pixel luminance "
                                 "variance exceeds this threshold use Floyd-Steinberg; "
                                 "sections below it use ordered dither. "
                                 "Lower = more FS sections.")

            # ── Tab 3 — Palette Solver ─────────────────────────────────────────
            with gr.Tab("Palette Solver"):
                gr.Markdown(
                    "The solver chooses which 9 BBC palette entries to change each "
                    "scanline to maximise coverage of the dithered colours. "
                    "The default greedy solver is fast (~0.5 s); "
                    "higher-quality options trade runtime for coverage.")

                with gr.Row():
                    with gr.Column():
                        solver = gr.Radio(
                            choices=["greedy", "z3"],
                            value="greedy",
                            label="Solver",
                            info="greedy — fast hill-climbing (default, < 1 s per image). "
                                 "z3 — exact SMT solver; slower but finds the optimal palette. "
                                 "Requires: pip install z3-solver")

                        changes = gr.Slider(
                            1, 16, value=CHANGE_PER_ROW, step=1,
                            label="Max palette changes per section",
                            info="Maximum palette slot changes allowed between sections "
                                 f"(default {CHANGE_PER_ROW}). Higher = more colour variety "
                                 "but uses more 6502 cycles; must match the 6502 raster code.")

                        look_ahead = gr.Checkbox(
                            value=False, label="2-step look-ahead",
                            info="Before committing to a palette change, check whether a "
                                 "follow-up change could unlock an additional colour. "
                                 "Only activates when direct gain is zero, preventing "
                                 "speculative bad choices. Helps colours needing two "
                                 "coordinated slots (e.g. white).")

                        restarts = gr.Slider(
                            1, 20, value=1, step=1,
                            label="Random restarts",
                            info="Run the greedy N times per section, each from a randomly "
                                 "perturbed starting palette. Keep the best result. "
                                 "5–20 trades runtime for quality; budget is always counted "
                                 "from the real previous palette.")

                    with gr.Column():
                        beam = gr.Slider(
                            1, 20, value=1, step=1,
                            label="Beam search width (1 = greedy)",
                            info="Keep the top-N palette candidates at each solver step "
                                 "instead of committing to one (beam search). "
                                 "Values 5–10 greatly reduce the chance of committing to a "
                                 "globally bad early choice, at N× the solver cost.")

                        anneal = gr.Slider(
                            0, 500, value=0, step=10,
                            label="Simulated annealing steps (0 = off)",
                            info="Replace the greedy with simulated annealing. Occasionally "
                                 "accepts palette changes that reduce coverage early on, "
                                 "allowing escape from local optima. Temperature cools "
                                 "exponentially over the step budget. 100–300 is typical. "
                                 "Overrides beam and restarts when > 0.")

                        smooth = gr.Slider(
                            0.0, 100.0, value=0.0, step=1.0,
                            label="Boundary smoothing penalty",
                            info="Cost subtracted from the greedy gain per new slot change "
                                 "(pixel-frequency units; 0 = off). Raises the bar for "
                                 "gratuitous palette changes, reducing visible horizontal "
                                 "banding at section boundaries. Values 5–30 are typical.")

            # ── Tab 4 — Output Options ─────────────────────────────────────────

        # ── Event wiring ──────────────────────────────────────────────────────

        # All controls that feed the live preprocessing preview
        _prep_controls = [
            input_img, resize,
            autolevel, input_gamma, contrast, brightness,
            saturation, hue, denoise, posterise, sharpen,
        ]

        for ctrl in _prep_controls:
            ctrl.change(
                fn=_apply_preprocessing,
                inputs=_prep_controls,
                outputs=prep_img,
            )

        # All controls fed to the full conversion
        _conv_controls = [
            input_img,
            # preprocessing
            resize, autolevel, input_gamma, contrast, brightness,
            saturation, hue, denoise, posterise, sharpen,
            # dithering
            dither, randomness, mixno, chunk_size, auto_threshold,
            # solver
            solver, changes, look_ahead, restarts, beam, anneal, smooth,
        ]

        emu_state = gr.State("")

        convert_btn.click(
            fn=_convert,
            inputs=_conv_controls,
            outputs=[conv_img, dl_file, dl_ssd, emu_state, status_box],
        ).then(
            fn=lambda html: html,
            inputs=[emu_state],
            outputs=[emu_link],
            show_progress="hidden",
        )

    return demo


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Base(),
    )
