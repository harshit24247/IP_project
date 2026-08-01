import torch
import torch.nn as nn
from model import GazeNet
import os

# Load model
model = GazeNet()
model.load_state_dict(
    torch.load('best_model.pth',
               map_location='cpu',
               weights_only=False)
)
model.eval()

# Export to ONNX
dummy = torch.randn(1, 1, 36, 60)

torch.onnx.export(
    model,
    dummy,
    'gaze_stm32.onnx',
    input_names   = ['eye_image'],
    output_names  = ['gaze_vector'],
    opset_version = 11
)

size = os.path.getsize('gaze_stm32.onnx') / 1024
print(f"✅ Exported successfully!")
print(f"📦 Model size: {size:.1f} KB")
print(f"📁 Saved as: gaze_stm32.onnx")