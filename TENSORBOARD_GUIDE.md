# 📊 TensorBoard Monitoring Guide

## Overview

TensorBoard is integrated into the training pipeline to provide real-time visualization of training progress. All models automatically log metrics, graphs, and histograms during training.

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

**1. Start TensorBoard in Docker container:**
```bash
docker exec -d facial-emotion-recognition-gpu tensorboard --logdir=/workspace/runs/tensorboard --host=0.0.0.0 --port=6006
```

**2. Open in browser:**
```
http://localhost:6006
```

**3. To stop TensorBoard:**
```bash
docker exec facial-emotion-recognition-gpu pkill tensorboard
```

### Option 2: Local Environment

**1. Open terminal in project directory**

**2. Activate virtual environment:**
```bash
.venv\Scripts\activate
```

**3. Start TensorBoard:**
```bash
tensorboard --logdir=runs/tensorboard --port=6006
```

**4. Open in browser:**
```
http://localhost:6006
```

**5. Press Ctrl+C to stop**

---

## 📈 What You Can Monitor

### 1. **Scalars Tab**
- **Training Accuracy**: Model accuracy on training data per epoch
- **Validation Accuracy**: Model accuracy on validation data per epoch
- **Training Loss**: Loss function value during training
- **Validation Loss**: Loss function value on validation set
- **Learning Rate**: Learning rate schedule over time

**Use Cases:**
- Detect overfitting (train acc ↑, val acc ↓)
- Monitor convergence
- Compare learning rates across models
- Identify when to stop training

### 2. **Graphs Tab**
- **Model Architecture**: Visual representation of model layers
- **Computational Graph**: TensorFlow operations graph

**Use Cases:**
- Understand model structure
- Debug architecture issues
- Compare model complexity

### 3. **Distributions Tab**
- **Weight Distributions**: How weights evolve during training
- **Bias Distributions**: Bias parameters over time
- **Activation Distributions**: Layer outputs

**Use Cases:**
- Detect vanishing/exploding gradients
- Monitor weight initialization
- Identify dead neurons

### 4. **Histograms Tab**
- **3D visualization** of distributions over time
- Similar to Distributions but with depth dimension

**Use Cases:**
- Detailed weight evolution analysis
- Compare layers across epochs

---

## 🔍 Advanced Usage

### Compare All Models Side-by-Side

TensorBoard automatically groups models when they're in the same root directory:

```
runs/tensorboard/
├── baseline_cnn/
├── deep_regularized/
├── vgg16/
├── resnet50v2/
├── efficientnet/
└── complex_cnn/
```

**In TensorBoard:**
1. Go to **Scalars** tab
2. Use **Show data download links** to export CSV
3. Use **Toggle All Runs** to show/hide models
4. Use **Smoothing slider** to reduce noise
5. Use **Runs selector** to filter specific models

### Filter Specific Metrics

Use the **regex filter** at the top:
- `.*accuracy.*` - Show only accuracy metrics
- `.*loss.*` - Show only loss metrics
- `val_.*` - Show only validation metrics

### Custom Grouping

Metrics are automatically grouped by prefix:
- `train/` - Training metrics
- `val/` - Validation metrics
- `test/` - Test metrics

---

## 📊 Interpreting Results

### Healthy Training
✅ Training loss decreases smoothly
✅ Validation loss follows training loss
✅ Gap between train/val accuracy is small (<10%)
✅ Learning rate decreases over time

### Overfitting Signs
⚠️ Training accuracy >> Validation accuracy
⚠️ Validation loss increases while training loss decreases
⚠️ Large gap (>15%) between train/val metrics

**Solutions:**
- Increase regularization (L2, Dropout)
- Add more data augmentation
- Reduce model complexity
- Use early stopping

### Underfitting Signs
⚠️ Both training and validation accuracy are low
⚠️ Loss plateaus early
⚠️ No improvement after many epochs

**Solutions:**
- Increase model complexity
- Train for more epochs
- Increase learning rate
- Remove excessive regularization

### Learning Rate Issues
⚠️ Loss oscillates wildly → Learning rate too high
⚠️ Loss decreases very slowly → Learning rate too low
⚠️ Sudden spikes in loss → Batch size too small or bad batches

---

## 🛠️ Troubleshooting

### Issue: TensorBoard not loading
**Solution:**
1. Check logs are being created: `ls runs/tensorboard/`
2. Verify TensorBoard is running: `docker ps` or check terminal
3. Try different port: `tensorboard --logdir=runs/tensorboard --port=6007`

### Issue: No data shown
**Solution:**
1. Wait for first epoch to complete
2. Refresh browser (Ctrl+F5)
3. Check logdir path is correct
4. Verify callbacks are included in training

### Issue: Port already in use
**Solution:**
```bash
# Find process using port 6006
netstat -ano | findstr :6006

# Kill the process (Windows)
taskkill /PID <PID> /F

# Or use different port
tensorboard --logdir=runs/tensorboard --port=6007
```

### Issue: Graphs not showing in Docker
**Solution:**
Ensure `write_graph=True` in TensorBoard callback:
```python
keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True  # ← Important!
)
```

---

## 💡 Tips & Best Practices

1. **Start TensorBoard before training** to monitor in real-time
2. **Use smoothing** (slider at top) to see trends more clearly
3. **Export data** for custom analysis using "Show data download links"
4. **Compare models** by selecting multiple runs in the sidebar
5. **Use regex filters** to focus on specific metrics
6. **Refresh regularly** during training to see latest updates
7. **Check histograms** if training seems unstable
8. **Monitor GPU utilization** in addition to TensorBoard metrics

---

## 📚 Resources

- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)
- [TensorBoard GitHub](https://github.com/tensorflow/tensorboard)
- [TensorBoard Tutorial](https://www.tensorflow.org/tensorboard/get_started)

---

**Happy Monitoring!** 📊🚀
