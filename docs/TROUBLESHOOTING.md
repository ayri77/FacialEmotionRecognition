# Troubleshooting Guide

## 🚨 Common Issues and Solutions

### 1. Model Loading Error (Keras Compatibility)

**Error Message**:
```
Could not deserialize class 'Functional' because its parent module keras.src.engine.functional cannot be imported
```

**Cause**:
The models were trained with Keras 2.x but the current environment uses Keras 3.x, which has breaking changes in model serialization.

**Solutions**:

#### ✅ **Immediate Solution** (Recommended)
- Use **Results Demo** mode - it's fully functional and shows all project achievements
- Click "📊 Switch to Results Demo" button in the Real-time Detection mode

#### 🔧 **Technical Solutions**:

1. **Use Original Docker Environment**:
   ```bash
   cd docker
   docker-compose up -d
   # Access Jupyter at http://localhost:8888
   ```

2. **Retrain Models** (Advanced):
   - Run the training notebooks with current Keras version
   - Models will be compatible with current environment

3. **Downgrade Keras** (Not Recommended):
   ```bash
   pip install keras==2.15.0
   # May cause other compatibility issues
   ```

### 2. Streamlit Not Found

**Error**: `streamlit: command not found`

**Solution**:
```bash
# Activate virtual environment first
.venv\Scripts\activate

# Then run streamlit
python -m streamlit run main_app.py
```

### 3. Import Errors

**Error**: `ModuleNotFoundError` or import issues

**Solution**:
```bash
# Ensure you're in the correct directory
cd web_app

# Activate virtual environment
.venv\Scripts\activate

# Check if all dependencies are installed
pip list | grep streamlit
```

### 4. File Not Found Errors

**Error**: `File does not exist: main_app.py`

**Solution**:
```bash
# Check current directory
pwd

# Navigate to correct directory
cd "C:\Users\pbori\Documents\Coureses\MIT\Projects\Capstone Projects\FacialEmotionRecognition\web_app"

# List files to verify
dir
```

## 🎯 Best Practices

### 1. Always Use Virtual Environment
```bash
# Before running any commands
.venv\Scripts\activate
```

### 2. Use Results Demo for Guaranteed Functionality
- Results Demo mode is fully functional
- Shows all project achievements
- No model loading issues

### 3. Check File Locations
- Ensure you're in the `web_app` directory
- Verify all files exist before running

### 4. Use Batch Files for Easy Launch
```bash
# Double-click or run
run_app.bat
```

## 🔍 Debugging Steps

### Step 1: Verify Environment
```bash
python -c "import sys; print('Python:', sys.executable)"
python -c "import streamlit; print('Streamlit version:', streamlit.__version__)"
```

### Step 2: Check File Structure
```bash
# Should show main_app.py, demo_results.py, app.py
dir web_app
```

### Step 3: Test Imports
```bash
cd web_app
python -c "import main_app; print('Main app OK')"
python -c "import demo_results; print('Demo results OK')"
```

### Step 4: Run Application
```bash
python -m streamlit run main_app.py
```

## 📞 Getting Help

### If Issues Persist:

1. **Check this guide first**
2. **Use Results Demo mode** (guaranteed to work)
3. **Verify virtual environment** is activated
4. **Check file locations** and permissions
5. **Use batch files** for easy launch

### Expected Behavior:

- ✅ **Results Demo**: Always works
- ⚠️ **Real-time Detection**: May have model loading issues (expected)
- ✅ **About**: Always works

## 🎉 Success Indicators

You'll know everything is working when:
- Application opens in browser at `http://localhost:8501`
- You can switch between modes using the sidebar
- Results Demo shows interactive charts and data
- No error messages in the terminal

Remember: **Results Demo mode is the recommended way to explore the project results!**
