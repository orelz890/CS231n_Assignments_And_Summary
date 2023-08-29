<div align="center">
  <h1>🔬 CS231n & CV7062610: Assignment Solutions & Summary 📚</h1>
  <p><b>Computer Vision Concepts Summary 📷</b></p>
  <p><i>Stanford & Ariel - Spring 2023 🌱</i></p>
</div>

<br/>

### 🌐 Overview

Welcome to the repository containing comprehensive summaries and assignment solutions for **CS231n** at Stanford University (2017) and **CV7062610** at Ariel University (2023). This collection represents a synthesis of my personal insights gained from these courses, in combination with supplementary materials sourced from the internet. **Please note that the summary is presented in a blend of English and Hebrew**. It's important to emphasize that no instructor has reviewed or endorsed this summary. The information provided is founded on my own interpretations.

<br/><br/>

<div align="center">
  <h1>📝 Assignment Solutions</h1>
</div>

<br/>

### Assignment Instructions 🗣️:

- Each assignment folder contains its instructions and my solution.

<br/>

### How To Run 🏃‍♂️:

- It is advised to run in [Colab](https://colab.research.google.com/), however, you can also run locally. To do so, first, set up your environment - either through [conda](https://docs.conda.io/en/latest/) or [venv](https://docs.python.org/3/library/venv.html). It is advised to install [PyTorch](https://pytorch.org/get-started/locally/) in advance with GPU acceleration. Then, follow the steps:


  1. Change every first code cell in `.ipynb` files to:
     ```bash
     %cd CV7062610/datasets/
     !bash get_datasets.sh
     %cd ../../
     ```
  2. Change the first code cell in section **Fast Layers** in [ConvolutionalNetworks.ipynb](assignment2/ConvolutionalNetworks.ipynb) to:
     ```bash
     %cd CV7062610
     !python setup.py build_ext --inplace
     %cd ..
     ```

Additionally, install the requirements specified under each assignment folder.

<br/><br/>

<div align="center">
  <h1>📚 Summary</h1>
</div>

<br/>

### 📖 Key Resources

- **Stanford Lecture videos (2017)** - [📺 Lecture Videos](https://www.youtube.com/playlist?list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk)
- **Ariel Lecture Slides (2023)** - Only Students can access that info...
<br/><br/>



<!-- Table of contents -->
### 🗺️🧭🔍 Summary Table of Contents
   * [01. Linear Classifier & Cross Validation](#image-0)
   * [02. Batch Normaliztion (More Details in 15)](#image-1)
   * [03. Optimizations - SGD, Momentun, Nesterov, Adam & Dropout (More Details in 17)](#image-2)
   * [04. Softmax & SVM](#image-7)
   * [05. Analitic Gradient](#image-8)
   * [06. Gradient Descent & Stochastic Gradient Descent](#image-10)
   * [17. Image Features & ConvNets](#image-12)
   * [18. Neural Networks](#image-13)
   * [19. Activation Functions](#image-14)
   * [10. Fully Connected Layer](#image-16)
   * [11. 2-Layer Neural Networks & How To Compute Gradients](#image-17)
   * [12. Convolution Layer](#image-18)
   * [13. Pooling Layer](#image-20)
   * [14. Weight Initialization - Xavier, Kaiming/MSRA](#image-22)
   * [15. Batch Normaliztion (More detailed)](#image-23)
   * [16. Transfer Learning](#image-26)
   * [17. Optimizations (More details)](#image-27)
   * [18. Enhancing CNN Robustness and Generalization - Data Augmentation, Fraction Pooling, Stochstic Depth, Cout & Mixup](#image-33)
   * [19. CNN Architectures - AlexNet, VGGNet, GoogleNet & ResNet](#image-35)
   * [20. GPT](#image-39)
   * [21. DenseNet & Neural Architecture Search(NAS)](#image-40)
   * [22. Practical Learning - Supervised, Unsupervised & Generative Modeling](#image-41)
   * [23. FVBN - PixelRNN & PixelCNN](#image-42)
   * [24. Autoencoders](#image-43)
   * [25. Generative Adversial Networks (GAN's)](#image-45)
   * [26. Pretext Tasks From Image Transformations](#image-48)
   * [27. SimCLR, Moco & CPC](#image-50)
   * [28. Generative Pre-Trained **Transformers** (GPT)](#image-52)
   * [29. Detection & Segmentation](#image-55)
   * [30. Computer Vision & Image Processing - **Sensors**](#image-57)
<br/><br/>



<!-- Actual Summary -->
### 📄 Summary
<div style="display: flex; flex-direction: column; align-items: center;">
    <img src="images/ff3d5f54_1.png" alt="Image 1" style="width: 95%; margin-bottom: 10px;">
    <img src="images/ff3d5f54_2.png" alt="Image 2" style="width: 95%; margin-bottom: 10px;">
    <img src="images/ff3d5f54_3.png" alt="Image 3" style="width: 95%; margin-bottom: 10px;">
    <img src="images/ff3d5f54_4.png" alt="Image 4" style="width: 95%; margin-bottom: 10px;">
    <img src="images/ff3d5f54_5.png" alt="Image 5" style="width: 95%; margin-bottom: 10px;">
    <img src="images/ff3d5f54_6.png" alt="Image 6" style="width: 95%; margin-bottom: 10px;">
    <img src="images/ff3d5f54_7.png" alt="Image 7" style="width: 95%; margin-bottom: 10px;">
    <img src="images/ff3d5f54_8.png" alt="Image 8" style="width: 95%; margin-bottom: 10px;">
    <img src="images/ff3d5f54_9.png" alt="Image 9" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_10.png" alt="Image 10" style="width: 95%; margin-bottom: 10px">
  <img src="images/ff3d5f54_11.png" alt="Image 11" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_12.png" alt="Image 12" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_13.png" alt="Image 13" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_14.png" alt="Image 14" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_15.png" alt="Image 15" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_16.png" alt="Image 16" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_17.png" alt="Image 17" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_18.png" alt="Image 18" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_19.png" alt="Image 19" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_20.png" alt="Image 20" style="width: 95%; margin-bottom: 10px">
  <img src="images/ff3d5f54_21.png" alt="Image 21" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_22.png" alt="Image 22" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_23.png" alt="Image 23" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_24.png" alt="Image 24" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_25.png" alt="Image 25" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_26.png" alt="Image 26" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_27.png" alt="Image 27" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_28.png" alt="Image 28" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_29.png" alt="Image 29" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_30.png" alt="Image 30" style="width: 95%; margin-bottom: 10px">
  <img src="images/ff3d5f54_31.png" alt="Image 31" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_32.png" alt="Image 32" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_33.png" alt="Image 33" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_34.png" alt="Image 34" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_35.png" alt="Image 35" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_36.png" alt="Image 36" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_37.png" alt="Image 37" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_38.png" alt="Image 38" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_39.png" alt="Image 39" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_40.png" alt="Image 40" style="width: 95%; margin-bottom: 10px">
  <img src="images/ff3d5f54_41.png" alt="Image 41" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_42.png" alt="Image 42" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_43.png" alt="Image 43" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_44.png" alt="Image 44" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_45.png" alt="Image 45" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_46.png" alt="Image 46" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_47.png" alt="Image 47" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_48.png" alt="Image 48" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_49.png" alt="Image 49" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_50.png" alt="Image 50" style="width: 95%; margin-bottom: 10px">
  <img src="images/ff3d5f54_51.png" alt="Image 51" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_52.png" alt="Image 52" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_53.png" alt="Image 53" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_54.png" alt="Image 54" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_55.png" alt="Image 55" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_56.png" alt="Image 56" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_57.png" alt="Image 57" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_58.png" alt="Image 58" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_59.png" alt="Image 59" style="width: 95%; margin-bottom: 10px;">
  <img src="images/ff3d5f54_60.png" alt="Image 60" style="width: 95%; margin-bottom: 10px">
  <img src="images/ff3d5f54_61.png" alt="Image 61" style="width: 95%; margin-bottom: 10px">
  <img src="images/ff3d5f54_62.png" alt="Image 62" style="width: 95%; margin-bottom: 10px">
  <img src="images/ff3d5f54_63.png" alt="Image 63" style="width: 95%; margin-bottom: 10px">
  <img src="images/ff3d5f54_64.png" alt="Image 64" style="width: 95%; margin-bottom: 10px">
  <img src="images/ff3d5f54_65.png" alt="Image 65" style="width: 95%; margin-bottom: 10px">
</div>

