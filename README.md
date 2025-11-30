# Face Mask Detection using VGG16 (facemask-dnn)

This repository contains a Colab/Notebook implementation for binary face mask detection using a transfer-learning approach built on top of VGG16. The primary experiment is available in the notebook:
- vgg16_mask_detection.ipynb
  https://github.com/hasnatsakil/facemask-dnn/blob/823c9d588e3b7073afeb150ae8c6a0678ac5187b/vgg16_mask_detection.ipynb

Project goal
- Train a simple classifier on top of a pretrained VGG16 (imagenet weights without top layers) to distinguish "Mask" vs "No Mask".
- Use data augmentation and a small dataset to achieve reasonable accuracy with minimal fine-tuning.

Dataset
- The notebook uses the "facemask-dataset" from Kaggle (sumansid/facemask-dataset):
  https://www.kaggle.com/datasets/sumansid/facemask-dataset
- The notebook demonstrates how to download the Kaggle dataset in Colab by uploading your `kaggle.json` credentials and running the Kaggle CLI.

Notebook overview
- Download and unzip the dataset in Colab.
- Inspect class distribution and split data into train/test folders.
- Use Keras' ImageDataGenerator (with augmentation) and validation split.
- Load VGG16 (include_top=False), freeze base layers, add Flatten -> Dense(256, relu) -> Dense(1, sigmoid).
- Compile with Adam, binary_crossentropy and train for configurable epochs.
- Plot training/validation accuracy and loss.

Requirements
- Python 3.8+
- TensorFlow (tested with TF 2.x)
- Keras (as included with TF 2.x)
- numpy, matplotlib, Pillow, kaggle (for Colab dataset download)
- Optional: Google Colab runtime for GPU acceleration

Quick start (Google Colab)
1. Open the notebook: vgg16_mask_detection.ipynb in Colab.
2. Upload your `kaggle.json` (from Kaggle account) to the Colab session.
3. Run the dataset download and unzip cells:
   - pip install kaggle
   - mkdir ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   - kaggle datasets download -d sumansid/facemask-dataset
   - unzip facemask-dataset.zip
4. Run the rest of the notebook cells. Set runtime > Change runtime type > GPU for faster training.

Key configurable parameters (in the notebook)
- image_size = (150,150)
- batch_size = 10
- number_epochs = 10
- learningRate = 0.001
- VGG16 weights: imagenet; vgg_model.trainable = False (frozen base)

Notes and tips
- The notebook uses featurewise centering / featurewise std normalization in ImageDataGenerator — ensure you call .fit() on the generator if you want to use featurewise normalization properly, or remove those options.
- With small datasets, heavy augmentation helps reduce overfitting, but be careful with excessive augmentation (e.g., channel_shift_range, vertical_flip might be unrealistic).
- Consider fine-tuning some top VGG16 layers (unfreeze last block) for better accuracy after initial training.
- Increase image size and batch size when using a GPU with more memory for potentially better results.
- Save model checkpoints and best model weights (ModelCheckpoint) to resume or deploy.

Results (from the example run in the notebook)
- Training accuracy increased to ~93% on the training subset in the notebook run, with validation accuracy fluctuating around 70–93% across epochs (dependent on random seed and split).

License & attribution
- Dataset: See Kaggle dataset page for usage terms.
- Model/training code: MIT-style / open — include your preferred license in the repo if needed.
- Acknowledgements: Uses pre-trained VGG16 weights from TensorFlow / Keras; dataset by sumansid on Kaggle.

Contributing
- Improvements welcome: better preprocessing, cross-validation, more robust train/test splitting, fine-tuning, or conversion to a reusable Python script.
- Please open an issue or PR with suggested changes or tests.

If you'd like, I can:
- Add a script to run training outside Colab (train.py).
- Add a model checkpointing cell and an export cell to save the final model (.h5 or SavedModel).
- Refactor the notebook to separate data preparation, model building, training, and evaluation for clarity.
