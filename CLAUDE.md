# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Converts images to BBC Master Mode 1 screen data with per-scanline palette changes. The system has two main parts:

1. **Python image converter** (`palsearch/`) — takes a PNG/JPG, preprocesses it, solves for optimal BBC palette assignments per 2-scanline section, dithers to BBC colours, and outputs a `.bin` file (palette + screen data).
2. **6502 assembly display engine** (`raster-fx.asm`) — runs on BBC Master hardware to render the image with raster-timed palette swaps each scanline.

## Build Commands

### Assemble the 6502 display engine
```
bin\beebasm.exe -i raster-fx.asm -do raster-fx.ssd -boot MyFX -v > compile.txt
```
Or simply: `make.bat`

### Run the Gradio web UI
```
pip install -r requirements.txt
python app.py
```
Opens at http://localhost:7860. The UI provides live preprocessing preview and full conversion with .bin/.ssd download.

### CLI conversion
```
python palsearch/palsearch.py input.png -o output.bin [-d ordered|fs|bn|auto] [-q]
```

### Preview a .bin file
```
python palsearch/showbin.py output.bin [-o preview.png]
python palsearch/showbin.py pal.bin pic.bin [-o preview.png]
```

### Split combined .bin into pal + pic
```
python bin/split_bin.py input.bin pal.bin pic.bin --split 1168
```

## Architecture

### Binary output format
- **Palette data** (variable size): 16-byte initial palette + 240 bytes padding (page-aligned for 6502) + delta streams (`changes_per_row × num_sections` bytes)
- **Screen data** (20480 bytes): BBC Mode 1 non-linear interleaved layout (8 rows interleaved byte-by-byte within each 640-byte block)
- Default: 9 palette changes per 2-scanline section, producing 1408 + 20480 = 21888 bytes total

### Palette byte encoding
`byte = (slot_index << 4) | (bbc_colour ^ 7)` — XOR 7 matches ULA negative logic.

### Key constants (palsearch.py)
- `SCREEN_W=320, SCREEN_H=256` — Mode 1 resolution
- `CHUNKSIZE=2` — scanlines per palette section (configurable to 1)
- `CHANGE_PER_ROW=9` — max palette slot changes per section (must match 6502 code)

### Solver options
- **greedy** (default) — fast hill-climbing with optional beam search, random restarts, simulated annealing, look-ahead
- **z3** — exact SMT solver (requires `pip install z3-solver`)

### Gradio UI (`palsearch/gradio_ui.py`)
- Preprocessing controls update a live preview (no solver run)
- "Convert" button runs `process_image()` from palsearch.py
- Can patch the output into the SSD template and generate a jsbeeb emulator link

### 6502 assembly (`raster-fx.asm`)
- Assembled with `beebasm.exe` (included in `bin/`)
- `CHUNK_SIZE=2` must match the Python converter's chunk size
- Performs cycle-counted raster palette swaps during display
