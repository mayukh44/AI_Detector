# AI Image Detector — AI Part

A PyTorch-based binary image classifier that predicts whether an image is
**AI-generated** or **REAL**, using transfer learning on a pretrained CNN
(ResNet18 by default).

This project contains only the AI/model side. MongoDB, MERN, frontend, and
backend integration are not required at this stage — they can be added later
once the model works, by wrapping `predict.py` in an API endpoint.

## File Structure

```
AI-Image-Detector/
├── dataset/
│   ├── ai/              # AI-generated images (label = 1)
│   └── real/             # Real/genuine images (label = 0)
├── test_images/           # Unseen images used to manually test predict.py
├── models/
│   └── ai_detector.pth    # Saved trained model (created after training)
├── src/
│   ├── preprocessing.py   # Image resize/normalize transforms
│   ├── dataset.py         # PyTorch Dataset that loads ai/ and real/
│   ├── model.py           # Model definition (transfer learning)
│   ├── train.py           # Training loop, saves ai_detector.pth
│   ├── evaluate.py        # Evaluates accuracy/metrics on a held-out split
│   └── predict.py         # Classifies a single new image
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Add your data

Put labeled images into:

```
dataset/ai/    -> images you know are AI-generated
dataset/real/  -> images you know are real/genuine
```

You decide the label simply by which folder you place each image in.

## 2. Train the model

```bash
cd src
python train.py
```

This will:
- Load images from `dataset/ai/` and `dataset/real/`
- Split them into train/validation sets
- Fine-tune a pretrained ResNet18
- Save the trained weights to `models/ai_detector.pth`

Key options (edit the `CONFIG` dict at the top of `train.py`):
- `epochs`, `batch_size`, `learning_rate`
- `img_size` (default 224x224, matches ResNet18 input)

## 3. Evaluate the model

```bash
python evaluate.py
```

Reports accuracy, precision, recall, F1-score, and a confusion matrix on the
validation split.

## 4. Predict on a new image

```bash
python predict.py --image ../test_images/test1.jpg
```

Outputs something like:

```
Prediction: AI-GENERATED
Confidence: 92.4%
```

You can also import `predict_image()` directly from `predict.py` in other
Python code (e.g. later, in a backend API route).

## Notes

- Images in `test_images/` should **never** be used for training — only for
  manual sanity checks of the trained model.
- `dataset.py` assigns labels automatically based on folder name:
  `AI = 1`, `REAL = 0`.
- The model uses transfer learning (a pretrained ImageNet backbone with a
  replaced final classification layer), so it needs comparatively little
  data and trains quickly even on CPU, though a GPU is recommended if you
  have more than a few thousand images.
