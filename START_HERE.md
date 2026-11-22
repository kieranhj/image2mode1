# image2mode1: start here

Converts full-colour images to BBC Micro MODE 1 with palette changes every 2 scanlines: a Python converter (palette solver + dithering, CLI and Gradio web UI) plus a cycle-timed 6502 display engine (`raster-fx.asm`). A tool with a demo viewer.

| | |
|---|---|
| GitHub | https://github.com/kieranhj/image2mode1 (public) |
| Local path | D:\beeb\mine\image2mode1 (moved from OneDrive\BEEB\Repos\image2mode1 on 2026-11-22) |
| Branch | master; other: wip/2026-11-tidy |
| Last commit | 2026-05-07: Add Spectrum test pics. |

## What's in here
- `palsearch/` - the converter: `palsearch.py` (CLI), `gradio_ui.py`, `showbin.py` (preview a .bin).
- `app.py` - Gradio web UI entry point (localhost:7860); `requirements.txt`.
- `raster-fx.asm` + `lib/` - the 6502 display engine (Model B and Master); `raster-fx.ssd` (tracked build), `testpal.bin` / `testpic.bin` (the image it shows).
- `bin/` - beebasm.exe, lz4.exe, split_bin.py.
- `pics/`, `spectrum/` - test images (`--spectrum` preset for pixel-art sources).
- `progress.md` - progress snapshot; `CLAUDE.md` - notes incl. the sync-constraint warning.

## Build and run
- `make`: runs `make.bat` -> `bin\beebasm.exe` assembles `raster-fx.asm` into `raster-fx.ssd`, listing in `compile.txt`. `make help` lists targets.
- Tested 2026-11-22 on a scratch copy: works, and `raster-fx.ssd` comes out byte-identical. Not run in place, because it rewrites `compile.txt`, which has uncommitted changes (and is tracked: consider gitignoring it).
- Converter: `py -3 -m pip install -r requirements.txt`, then `py -3 app.py` (web UI) or `py -3 palsearch\palsearch.py input.png -o output.bin`.
- Run: boot `raster-fx.ssd` in an emulator or on a BBC; the web UI can also make a jsbeeb link (README still says bbc.godbolt.org; bbc.xania.org is the current host).

## Local changes
- Uncommitted now: `compile.txt` (rebuilt listing), untracked `spectrum/skyline.png`.
- `wip/2026-11-tidy` (local + origin) snapshots exactly those two as of 2026-11-21.

## Status and next steps
- progress.md: "ready to share"; converter work plateaued, later work was raster timing, Model B support and the README. 2026-05 added `--seed-mode`, `--solve-order`, the `--spectrum` preset and a 'paste' resize mode.
- No open TODO list found beyond progress.md.

## See also
- README.md (full CLI options, binary format, how it works), CLAUDE.md, progress.md.
- D:\beeb\mine\image2mode7 (the MODE 7 sibling).
