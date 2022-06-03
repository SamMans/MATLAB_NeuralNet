# MATLAB_NeuralNet
This small MATLAB package can help tutors and machine learning beginners reflect on the effect of varying different training parameters on the network performance. The package is highly recommended for both machine learning beginners and tutors.

## System Requirements and Installations
- MATLAB R2022a

## Dependencies
- None

## How to run?
### Start a new project
- Create a folder with your project name in the package directory
- The folder must contain two other folders: dataset and networks
- The dataset folder must include three .mat files, namely: train.mat, cross_v.mat and test.mat
- The train.mat file must include three variables: X (input features size: number of experiments * number of features), y (output size: number of experiments * 1) and num_labels (number of output labels)
- The cross_v.mat must also contain two variables: Xcv (validation features size: number of validation experiments * number of features), ycv (validation output size: number of validation experiments * 1)
- The test.mat must also contain two variables: Xtest (test features size: number of test experiments * number of features), ycv (test output size: number of test experiments * 1)
- You needn't add anything to the networks folder, this will automatically save the networks you create using the package
- For the sake of illustration, a hand-digit recognition dataset from Andrew's machine learning course on Coursera is added to a project folder called "char_class" 
  
### Open NN_make.mlx live script
- Choose network parameters and project folder name
- Click "Build" to initialize your network parameters
- Select your training parameters
- Click "Train" to train your network and run diagnostics
- Click "Test" to test your network using the testing dataset
- You can click "Save" to save your network object in the "networks" directory of the project 
