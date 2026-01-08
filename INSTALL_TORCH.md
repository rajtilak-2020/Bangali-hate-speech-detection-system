# Installing PyTorch - Windows Long Path Issue

## Problem
PyTorch installation is failing due to Windows Long Path support not being enabled. The error message indicates:
```
ERROR: Could not install packages due to an OSError: [Errno 2] No such file or directory
```

## Solution Options

### Option 1: Enable Windows Long Path Support (Recommended)

1. **Open Registry Editor** (Press `Win + R`, type `regedit`, press Enter)
2. Navigate to: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
3. Find the key: `LongPathsEnabled`
4. Set its value to `1` (if it doesn't exist, create it as DWORD)
5. **Restart your computer**
6. Try installing again: `pip install torch transformers datasets accelerate`

### Option 2: Install PyTorch using Conda (Alternative)

If you have Anaconda or Miniconda installed:
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install transformers datasets accelerate
```

### Option 3: Install PyTorch CPU-only version

Try installing the CPU-only version which might have shorter paths:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets accelerate
```

### Option 4: Use a Virtual Environment in a Shorter Path

Create a virtual environment in a shorter path (like `C:\venv`):
```bash
python -m venv C:\venv
C:\venv\Scripts\activate
pip install -r requirements.txt
```

## Current Status

✅ **Installed:**
- flask
- pandas
- scikit-learn
- numpy
- scipy

❌ **Still Need:**
- torch
- transformers
- datasets
- accelerate

## After Installing PyTorch

Once PyTorch is installed, run:
```bash
pip install transformers datasets accelerate
```

Then you can proceed with training:
```bash
python train_model.py
```

