


## Requierments & installations:
1. Have the latest version of Anaconda installed on your machine.
https://linuxhint.com/install-anaconda-ubuntu-22-04/

2. install torch:
https://installati.one/install-python3-torch-ubuntu-22-04/

3. install: pandas torchvision matplotlib

4. Get the dataset:
In order for this code to work you need to extract the dataset from this link here:
https://drive.google.com/file/d/1eSp5f5ih17blcqjgxJQ1IKx9a7QXTqJT/view?usp=sharing

The folder name of the dataset should be called - Rice_Image_dataset
Copy paste it into the project folder and please **dont change its name!**



## How to tain the model:
After you are all done with the setup, Just run the model_creator.py

## How to use the pretrained model example:
python3 myScript.py "Rice_Image_Dataset/Ipsala/Ipsala (2).jpg"



## Network Architecture:

The model architecture consists of three sets of **convolutional layers** followed by **batch normalization**, **ReLU activation**, and **max pooling** 3 times. After the convolutional layers, the output is flattened and passed through a **dropout** layer with a dropout rate of 0.2. Finally, a fully connected layer maps the flattened features to the output classes.

I used the **Adam** optimizer. The Adam optimizer combines the concepts of adaptive learning rates and momentum, which helps in efficient gradient-based optimization.

### Structure
3 * [CONV >> Batch Normalization >> Relu Activation >> Max Pooling] >> Adaptive Avrage Pooling >> Dropout >> Fully Connected


In our case, the size of the input tensor after the convolutional layers was [batch_size, out_channels_3, 4, 4]. However, the subsequent linear layer expected the input tensor to have a different size ([batch_size, out_channels_3 * 4 * 4]) By adding the nn.AdaptiveAvgPool2d layer with an output size of (4, 4), we effectively reduced the spatial dimensions of the feature maps to match the expected size. 

Those are the filters I tryied:

Filters: 5x5, 3x3

Num of filters: 16, 32, 64
