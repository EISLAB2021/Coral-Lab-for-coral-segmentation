# Coral-Lab-for-coral-segmentation
Coral-Lab is an improved semantic segmentation model based on DeepLabV3+, which is designed for coral identification task.

# Basaic information
1. Necessary environment is saved in requirement.txt
2. Pytorch is GPU version, can be downloaded in https://pytorch.org/
3. The main body of coeds include ‘config.py’, ‘data_loader.py’, ‘image_saver.py’, ‘Loss_Function.py’, ‘model_generation.py’, ‘inference.py’ and ‘train.py’.
4. Tools for debugging and assessment include ‘Assessment.py’ and ‘Generation_train_TXT.py’.

# Introduction of main body of codes
1)	config.py: includes definitions of the parameters, helping us run the code with those parameters.
2)	data_loader.py: is used by train.py and inference.py to load data, and the data augmentation code is contained in this file.
3)	image_saver.py: is used to save the images generated during training (including fused images and mask images), and it also saves the inference results.
4)	Loss_Function.py: contains all the potential loss functions, and after testing, the composite Focal–Dice loss achieved the best performance on the Kumejima dataset.
5)	model_generation.py: contains the code for loading and constructing the model and is used by train.py and inference.py.
6)	inference.py: load trained model for inference task.
7)	train.py: load training dataset and evaluation dataset to train semantic segmentation models.

# Introduction of tools
1)	Assessment.py: using test dataset to verify the performance of trained model
2)	Generation_train_TXT.py: generate .txt files of training dataset and evaluation dataset for ‘train.py’. It’s noted that this script will overwrite existing files and the training dataset and evaluation dataset are generated randomly every time. So, if you don’t want to change training and evaluation dataset, please don’t run this code. 

# How to use it to estimate
Inference task: The project folder includes 2 folders named as ‘InferenceIn’ and ‘InferenceOut’ respectively. Put images in ‘InferenceIn’ and use following config to run ‘inference.py’:
`--data_dir ./InferenceIn/Sample --out_dir ./InferenceOut/Sample --model_name corallab --data_size 512 --segmentation_only --model_dir ./checkpoint/corallab34.pt`

`./InferenceIn/Sample` and `./InferenceOut/Sample` can be changed to your path
