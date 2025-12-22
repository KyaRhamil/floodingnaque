# 🪟 Windows Installation Guide - Floodingnaque Backend

## 🚨 Issue: Build Errors on Windows

If you're getting errors like "Compiler cl cannot compile programs" when installing pandas, numpy, or other packages, this is because some packages need to be compiled from source, which requires Visual Studio C++ Build Tools on Windows.

## ✅ **Solution: Use Pre-Built Wheels**

### **Option 1: Install Minimal Requirements (RECOMMENDED)**

Use the minimal requirements file which automatically gets pre-built wheels:

```powershell
# Make sure you're in the backend directory and venv is activated
cd backend
.\venv\Scripts\Activate.ps1

# Install minimal requirements (no build needed!)
pip install -r requirements-minimal.txt
```

This installs all **essential** packages needed to run the backend.

---

### **Option 2: Install Full Requirements (Updated)**

The updated `requirements.txt` now uses version ranges that should work better:

```powershell
# Try installing the full requirements
pip install -r requirements.txt
```

If you still get errors, try:

```powershell
# Install packages individually (pip will find pre-built wheels)
pip install Flask flask-cors Werkzeug
pip install requests gunicorn
pip install pandas numpy scikit-learn joblib
pip install APScheduler python-dotenv SQLAlchemy
```

---

### **Option 3: Install Visual Studio Build Tools (If needed)**

**Only if you need the exact versions** and want to build from source:

1. **Download Visual Studio Build Tools**:
   - Visit: https://visualstudio.microsoft.com/downloads/
   - Scroll down to "Tools for Visual Studio"
   - Download "Build Tools for Visual Studio 2022"

2. **Install with C++ workload**:
   - Run the installer
   - Select "Desktop development with C++"
   - Click Install (requires ~7GB)

3. **After installation, retry**:
   ```powershell
   pip install -r requirements.txt
   ```

---

## 🎯 Quick Setup (5 Minutes)

### **Step 1: Activate Virtual Environment**
```powershell
# From project root
.\venv\Scripts\Activate.ps1
cd backend
```

### **Step 2: Install Minimal Requirements**
```powershell
pip install -r requirements-minimal.txt
```

### **Step 3: Create .env File**
```powershell
# Copy example configuration
Copy-Item .env.example .env

# Edit .env and add your API keys
notepad .env
```

Add your actual API keys:
```env
OWM_API_KEY=your_actual_openweathermap_key
METEOSTAT_API_KEY=your_actual_weatherstack_key
DATABASE_URL=sqlite:///data/floodingnaque.db
```

### **Step 4: Verify Installation**
```powershell
# Check database
python scripts/inspect_db.py

# Start the server
python main.py
```

You should see:
```
Starting Floodingnaque API on 0.0.0.0:5000 (debug=False)
```

### **Step 5: Test the API**
Open another PowerShell window:
```powershell
# Health check
curl http://localhost:5000/health

# Or use browser
Start-Process "http://localhost:5000"
```

---

## 📦 What's Installed (Minimal)

### **Core Packages**
✅ Flask 3.0+ - Web framework
✅ flask-cors - CORS support
✅ requests - HTTP client
✅ SQLAlchemy 2.0+ - Database ORM
✅ python-dotenv - Environment variables

### **Machine Learning**
✅ pandas - Data processing
✅ numpy - Numerical computing
✅ scikit-learn - ML models
✅ joblib - Model serialization

### **Background Tasks**
✅ APScheduler - Scheduled jobs
✅ gunicorn - Production server

---

## 🔧 Install Optional Packages Later

If you need additional functionality, install these separately:

```powershell
# Visualization (for thesis charts)
pip install matplotlib seaborn

# Security enhancements
pip install cryptography validators bleach

# Rate limiting
pip install Flask-Limiter

# Database migrations
pip install alembic

# Testing
pip install pytest pytest-cov faker

# Development
pip install jupyter ipython
```

---

## 🆘 Troubleshooting

### **Error: "Microsoft Visual C++ 14.0 or greater is required"**
✅ Solution: Use `requirements-minimal.txt` (doesn't need compiler)

### **Error: "Compiler cl cannot compile programs"**
✅ Solution: Use version ranges (already fixed in requirements.txt)

### **Error: "No module named 'numpy'"**
```powershell
pip install numpy --no-build-isolation
```

### **Error: "pip is out of date"**
```powershell
python -m pip install --upgrade pip setuptools wheel
```

### **Virtual Environment Issues**
```powershell
# Deactivate and recreate
deactivate
Remove-Item -Recurse -Force venv
python -m venv venv
& venv/Scripts/Activate.ps1
pip install -r requirements-minimal.txt
```

---

## ✅ Verification Checklist

After installation, verify everything works:

```powershell
# Check Python version
python --version  # Should be 3.8+

# Check pip version
pip --version

# Check installed packages
pip list | Select-String "Flask|pandas|numpy|scikit"

# Verify database
python scripts/inspect_db.py

# Test imports
python -c "import flask, pandas, numpy, sklearn; print('All imports successful!')"

# Start server
python main.py
```

---

## 🎓 For Thesis Work

You can run the backend with minimal requirements. The optional packages are only needed for:

- **matplotlib/seaborn**: Generating charts (can do separately)
- **cryptography**: Advanced encryption (not critical for demo)
- **pytest**: Running tests (can add later)
- **jupyter**: Interactive development (optional)

**The backend API works perfectly with just the minimal requirements!**

---

## 💡 Pro Tips

1. **Use minimal requirements first** - Get the backend running quickly
2. **Install optional packages later** - Only when you need them
3. **Keep pip updated** - `python -m pip install --upgrade pip`
4. **Use PowerShell as Administrator** - For better permissions
5. **Check Python version** - Make sure it's 3.8 or higher

---

## 🚀 Next Steps

Once installed successfully:

1. ✅ Create your `.env` file with API keys
2. ✅ Start the server: `python main.py`
3. ✅ Test the API: `curl http://localhost:5000/health`
4. ✅ Review documentation: `QUICK_START_v2.md`
5. ✅ Train your model: `python scripts/train.py`

---

## 📞 Still Having Issues?

### **Quick Fix Command**
```powershell
# Nuclear option - fresh start (from project root)
Remove-Item -Recurse -Force venv
python -m venv venv
.\venv\Scripts\Activate.ps1
cd backend
pip install --upgrade pip setuptools wheel
pip install -r requirements-minimal.txt
```

This should work 99% of the time on Windows!

---

**Last Updated**: December 12, 2025
**Tested On**: Windows 11, Python 3.12
**Status**: ✅ Working
