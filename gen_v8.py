#!/usr/bin/env python3
"""Generate Version 8 notebook by assembling cell files."""
import json, os, glob

CELL_DIR = "/Users/rahulraj1406/ETsy/v8_cells"
os.makedirs(CELL_DIR, exist_ok=True)

def write_cell(idx, content):
    path = os.path.join(CELL_DIR, f"cell_{idx:02d}.py")
    with open(path, 'w') as f:
        f.write(content)
    return path

def build_notebook(cell_dir, output_path):
    files = sorted(glob.glob(os.path.join(cell_dir, "cell_*.py")))
    cells = []
    for fpath in files:
        with open(fpath) as f:
            source = f.read()
        lines = source.split("\n")
        src_list = [line + "\n" for line in lines[:-1]]
        if lines[-1]:
            src_list.append(lines[-1])
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": src_list,
            "execution_count": None,
            "outputs": []
        })
    
    notebook = {
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat_minor": 5,
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": cells
    }
    
    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    print(f"Generated: {output_path} ({len(cells)} cells)")

if __name__ == "__main__":
    build_notebook(CELL_DIR, "Version 8 - Triple Vision Ensemble.ipynb")
