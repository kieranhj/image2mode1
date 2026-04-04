# image2mode1

Convert full-colour images to BBC Master Mode 1 with per-scanline palette changes.

BBC Micro Mode 1 is limited to 4 colours from a palette of 8, but the ULA palette can be reprogrammed mid-frame. This project exploits cycle-counted raster timing to swap palette entries every 2 scanlines, producing images with far more apparent colour than the hardware nominally supports.

The system has two parts:

- **Python image converter** (`palsearch/`) — preprocesses an input image, solves for optimal palette assignments per 2-scanline section, dithers to BBC colours, and outputs a `.bin` file.
- **6502 assembly display engine** (`raster-fx.asm`) — runs on BBC Master hardware (or emulator) to render the image with raster-timed palette swaps each scanline.

## Quick start

### Install Python dependencies

```bash
pip install -r requirements.txt
```

### Web UI (recommended)

```bash
python app.py
```

Opens a Gradio interface at http://localhost:7860 with:

- Interactive preprocessing controls (saturation, contrast, brightness, gamma, etc.) with live preview
- Multiple dithering and solver options
- Download converted `.bin` or `.ssd` disk image
- Generate a [jsbeeb](https://bbc.godbolt.org/) emulator link to view the result in-browser

### Command-line conversion

```bash
python palsearch/palsearch.py input.png -o output.bin
```

### Preview a .bin file

```bash
python palsearch/showbin.py output.bin [-o preview.png]
python palsearch/showbin.py pal.bin pic.bin [-o preview.png]
```

### Build the 6502 display engine

```bash
bin\beebasm.exe -i raster-fx.asm -do raster-fx.ssd -boot MyFX -v > compile.txt
```

Or: `make.bat`

## CLI options

### Dithering (`-d`)

| Method | Flag | Description |
|--------|------|-------------|
| Ordered | `ordered` | Colour-aware 2x2 Bayer pattern (default) |
| Floyd-Steinberg | `fs` | Error diffusion with serpentine scanning |
| Blue noise | `bn` | 128x128 blue-noise texture for smooth gradients |
| None | `none` | Nearest-colour, no dithering |
| Auto | `auto` | Per-section: FS for high-variance areas, ordered for low-variance |

### Solver (`-s`)

| Solver | Description |
|--------|-------------|
| `greedy` | Fast hill-climbing (default). Supports beam search, random restarts, simulated annealing, and look-ahead. |
| `z3` | Exact SMT solver via z3-solver. Optimal results but significantly slower. |

### Image preprocessing

| Option | Default | Description |
|--------|---------|-------------|
| `--saturation` | 1.0 | Colour saturation multiplier (1.3-1.8 typical for photos) |
| `--contrast` | 1.0 | Contrast multiplier |
| `--brightness` | 1.0 | Brightness multiplier |
| `--input-gamma` | 1.0 | Gamma curve (>1 brightens) |
| `--hue` | 0 | Hue rotation in degrees |
| `--autolevel` | off | Histogram stretch for washed-out images |
| `--denoise` | off | 3x3 median filter |
| `--posterise` | 0 | Reduce to N bits per channel (1-7) |
| `--sharpen` | 0 | Pre-sharpen (0.5 mild, 1.0 strong) |

### Solver tuning

| Option | Default | Description |
|--------|---------|-------------|
| `--chunk-size` | 2 | Scanlines per palette section (1 or 2) |
| `--changes` | 9 | Max palette changes per section (must match 6502 code) |
| `--restarts` | 1 | Random restart attempts (5-20 for quality) |
| `--beam` | 1 | Beam search width (5-10 for better coverage) |
| `--anneal` | 0 | Simulated annealing steps (replaces greedy if >0) |
| `--look-ahead` | off | 2-step look-ahead for hard-to-reach colours |
| `--randomness` | 64 | Per-pixel dither bias (0-255) |
| `--smooth` | 0 | Min coverage improvement % per slot change (1-5 typical) |
| `--mixno` | 0 | Mixes table index (0 = smoothest) |

### Other

| Option | Description |
|--------|-------------|
| `-r`, `--resize` | Fit method: `fit`, `crop-left`, `crop-right`, `crop-top`, `crop-bottom` |
| `-p`, `--preview` | Write preview PNG of converted image |
| `-q`, `--quiet` | Suppress per-section progress output |

## Binary output format

The `.bin` file contains palette data followed by screen data:

| Offset | Size | Content |
|--------|------|---------|
| 0 | 16 | Initial 16-entry ULA palette |
| 16 | 240 | Padding (page-aligned for 6502 performance) |
| 256 | 1152 | Per-section palette delta streams (9 changes x 128 sections) |
| 1408 | 20480 | Mode 1 screen data (non-linear interleaved layout) |

**Total: 21888 bytes** (default settings)

### Palette byte encoding

```
byte = (slot_index << 4) | (bbc_colour XOR 7)
```

- `slot_index`: which of the 16 ULA palette entries to update (0-15)
- `bbc_colour`: 3-bit RGB (bit0=R, bit1=G, bit2=B), XOR 7 for ULA negative logic

## How it works

### The palette solver

Mode 1 displays 4 logical colours per pixel, mapped to physical colours via the ULA palette. The palette has 16 entries (4 logical colours x 4 entries each, due to the bit-pairing scheme). By changing palette entries between scanlines, different rows can display different physical colours.

The solver divides the 256-scanline display into sections (default: 2 scanlines each = 128 sections). For each section it determines which palette changes will best represent the source image colours, given a budget of 9 changes per section (constrained by the available CPU cycles in the horizontal blanking interval).

The greedy solver evaluates all 112 possible single-slot changes (16 slots x 7 colours) at each step and picks the change that maximises frequency-weighted coverage of required colour combinations. Optional enhancements (beam search, random restarts, simulated annealing, look-ahead) improve quality at the cost of speed.

### The display engine

The 6502 assembly code (`raster-fx.asm`) uses cycle-counted interrupt timing on the BBC Master to:

1. Synchronise to the vertical sync pulse
2. Set the initial palette
3. At each 2-scanline boundary, apply the palette delta stream — writing up to 9 new palette values to the ULA within the horizontal blanking interval

This requires sub-microsecond precision. The `CHUNK_SIZE` and `CHANGE_PER_ROW` constants in the assembly **must match** the Python converter settings, or the output will be garbled.

## Project structure

```
image2mode1/
├── app.py                  # Gradio web UI entry point
├── raster-fx.asm           # 6502 display engine
├── make.bat                # Assembly build script
├── requirements.txt        # Python dependencies
├── bin/
│   ├── beebasm.exe         # 6502 assembler
│   └── split_bin.py        # Split combined .bin into pal + pic
├── lib/                    # 6502 assembly libraries
├── palsearch/
│   ├── palsearch.py        # Image converter and palette solver
│   ├── gradio_ui.py        # Web UI implementation
│   ├── showbin.py          # .bin file previewer
│   └── bluenoise.png       # Blue-noise dither texture
└── pics/                   # Example input images
```

## Requirements

- **Python 3.8+** with `numpy`, `Pillow`, `gradio`
- **z3-solver** (optional, for exact SMT solving)
- **beebasm** (included in `bin/` for Windows)
- **BBC Master** or emulator (e.g. [jsbeeb](https://bbc.godbolt.org/), [b-em](https://github.com/stardot/b-em)) to run the output

## Utilities

### Split a combined .bin file

```bash
python bin/split_bin.py input.bin pal.bin pic.bin --split 1168
```

Separates the palette and screen data for use with alternative loaders.

## Author

Kieran Connell
