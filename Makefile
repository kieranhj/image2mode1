# Wrapper so that `make` works from PowerShell / Windows Terminal.
# GNU make runs recipes with cmd.exe here (no sh on PATH).
.PHONY: all build help

all: build

build:
	cmd /c .\make.bat

help:
	@echo make        - assemble raster-fx.asm into raster-fx.ssd (runs make.bat; listing in compile.txt)
	@echo make help   - this list
	@echo Converter: py -3 app.py (Gradio web UI on localhost:7860) or the palsearch CLI - see README.md
