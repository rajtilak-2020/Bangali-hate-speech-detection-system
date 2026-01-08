# Virtual Environment Usage Guide

## ✅ All packages have been successfully installed!

A virtual environment has been created at `C:\venv` to avoid Windows Long Path issues.

## How to Use the Virtual Environment

### To activate the virtual environment:

**In Command Prompt (CMD):**
```cmd
C:\venv\Scripts\activate
```

**In PowerShell:**
```powershell
C:\venv\Scripts\Activate.ps1
```

**Or use the Python directly:**
```cmd
C:\venv\Scripts\python.exe your_script.py
```

### To deactivate:
```cmd
deactivate
```

## Running Your Scripts

### 1. Train the Model:
```cmd
C:\venv\Scripts\python.exe train_model.py
```

### 2. Run the Web Interface:
```cmd
C:\venv\Scripts\python.exe app.py
```

Then open your browser and go to: `http://127.0.0.1:5000`

## Quick Commands

**Check installed packages:**
```cmd
C:\venv\Scripts\pip.exe list
```

**Install additional packages:**
```cmd
C:\venv\Scripts\pip.exe install package_name
```

**Update packages:**
```cmd
C:\venv\Scripts\pip.exe install --upgrade package_name
```

## Installed Packages

✅ torch (2.9.1)
✅ transformers (4.57.3)
✅ datasets (4.4.2)
✅ pandas (2.3.3)
✅ numpy (2.4.0)
✅ scikit-learn (1.8.0)
✅ accelerate (1.12.0)
✅ flask (3.1.2)
✅ And all dependencies

## Next Steps

1. **Train your model:**
   ```cmd
   C:\venv\Scripts\python.exe train_model.py
   ```

2. **Run the web interface:**
   ```cmd
   C:\venv\Scripts\python.exe app.py
   ```

3. **Open browser:** `http://127.0.0.1:5000`

Happy coding! 🚀

