import os
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as tnn
import torch.optim as toptim

try:
   from mmcv.ops import DeformConv2d as TorchDeformConv2d
except ImportError:
   print("mmcv not installed, skipping PyTorch training.")
   TorchDeformConv2d = None


class TorchDeformableCNN(tnn.Module):
   def __init__(self):
       super().__init__()
       self.offset = tnn.Conv2d(1, 18, kernel_size=3, padding=1)
       self.deform = TorchDeformConv2d(1, 8, kernel_size=3, padding=1)
       self.relu = tnn.ReLU()
       self.pool = tnn.MaxPool2d(2)
       self.fc = tnn.Linear(8 * 14 * 14, 10)

   def forward(self, x):
       offset = self.offset(x)
       x = self.relu(self.deform(x, offset))
       x = self.pool(x)
       x = x.view(x.size(0), -1)
       return self.fc(x)


import jittor as jt
from jittor import nn
from deform_conv import DeformConv2d

jt.flags.use_cuda = 0

class JittorDeformNet(nn.Module):
   def __init__(self):
       super().__init__()
       self.deform = DeformConv2d(1, 8, 3, padding=1)
       self.relu = nn.ReLU()
       self.pool = nn.Pool(2, 2)
       self.fc = nn.Linear(8 * 14 * 14, 10)

   def execute(self, x):
       x = self.relu(self.deform(x))
       x = self.pool(x)
       x = x.reshape([x.shape[0], -1])
       output = self.fc(x)
       if isinstance(output, tuple):
           return output[0]
       return output


def extract_tensor(data):
   while isinstance(data, tuple):
       data = data[0]
   return data


def load_model(model, model_path, framework="jittor"):
   if not os.path.exists(model_path):
       raise FileNotFoundError(f"Model file not exists: {model_path}")

   try:
       if framework == "jittor":
           model.load_state_dict(jt.load(model_path))
       elif framework == "pytorch":
           model.load_state_dict(torch.load(model_path, map_location="cpu"))
       print(f"Successfully loaded {framework} model: {model_path}")
       return model
   except Exception as e:
       raise RuntimeError(f"Model loading failed: {str(e)}")


def test(framework="jittor", model_path=None):
   data_dir = "data/sample_dataset"
   if not os.path.exists(data_dir):
       os.makedirs(data_dir, exist_ok=True)
       np.save(os.path.join(data_dir, "X.npy"), np.random.randn(100, 1, 28, 28).astype(np.float32))
       np.save(os.path.join(data_dir, "y.npy"), np.random.randint(0, 10, size=100))

   X_test = np.load(os.path.join(data_dir, "X.npy"))[:20]
   y_test = np.load(os.path.join(data_dir, "y.npy"))[:20]
   print(f"Test data loaded: {len(X_test)} samples")

   if framework == "jittor":
       model = JittorDeformNet()
       default_path = "compare_with_pytorch/jittor_model.pth"
   elif framework == "pytorch" and TorchDeformConv2d is not None:
       model = TorchDeformableCNN()
       default_path = "compare_with_pytorch/pytorch_model.pth"
   else:
       raise ValueError(f"Unsupported framework: {framework}")

   model_path = model_path or default_path
   model = load_model(model, model_path, framework)
   model.eval()

   start_time = time.time()
   correct = 0
   total = len(X_test)

   try:
       if framework == "jittor":
           with jt.no_grad():
               inputs = jt.array(X_test)
               outputs = model(inputs)
               outputs = extract_tensor(outputs)
               pred_classes = outputs.argmax(dim=1)
               pred_classes = extract_tensor(pred_classes)
               pred_np = pred_classes.numpy() if hasattr(pred_classes, 'numpy') else np.array(pred_classes)

       elif framework == "pytorch":
           with torch.no_grad():
               inputs = torch.tensor(X_test).float()
               outputs = model(inputs)
               pred_classes = outputs.argmax(dim=1)
               pred_np = pred_classes.numpy()

       correct = (pred_np == y_test).sum()
       accuracy = correct / total
       infer_time = time.time() - start_time

       print(f"\n===== {framework} Test Results =====")
       print(f"Test samples: {total}")
       print(f"Correct predictions: {correct}")
       print(f"Test accuracy: {accuracy:.2%}")
       print(f"Inference time: {infer_time:.4f}s")
       print(f"Average time per sample: {infer_time / total:.4f}s")
       return accuracy

   except Exception as e:
       print(f"Test failed: {str(e)}")
       return 0.0


if __name__ == "__main__":
   test(framework="jittor")
   if TorchDeformConv2d is not None:
       test(framework="pytorch")

   X_test = np.load("data/sample_dataset/X.npy")[:5]
   y_test = np.load("data/sample_dataset/y.npy")[:5]

   jt_model = JittorDeformNet()
   jt_model = load_model(jt_model, "compare_with_pytorch/jittor_model.pth", "jittor")
   jt_model.eval()

   with jt.no_grad():
       jt_inputs = jt.array(X_test)
       jt_outputs = jt_model(jt_inputs)
       jt_outputs = extract_tensor(jt_outputs)
       jt_preds_tensor = jt_outputs.argmax(dim=1)
       jt_preds_tensor = extract_tensor(jt_preds_tensor)
       jt_preds = jt_preds_tensor.numpy() if hasattr(jt_preds_tensor, 'numpy') else np.array(jt_preds_tensor)

   plt.figure(figsize=(10, 4))
   for i in range(5):
       plt.subplot(1, 5, i + 1)
       plt.imshow(X_test[i, 0], cmap="gray")
       plt.title(f"True: {y_test[i]}\nPred: {jt_preds[i]}")
       plt.axis("off")
   plt.tight_layout()
   plt.savefig("test_predictions.png")
   print("\nPrediction visualization saved to: test_predictions.png")
