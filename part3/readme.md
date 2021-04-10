# Part-3
## Note

1. Before proceeding go to this link: [This link](https://drive.google.com/drive/folders/1_JHcB4aThYCxZXwpEFHgwYVPpfZjZvOi?usp=sharing)
2. Then download the file named **important_part3.zip**
3. Extract all files and folders in this directory

### Run all experiments
 Run **task2_3_final.ipynb** this to reproduce all results

### To Test model that was trained using pretrained network

Create a directory named **test_loader** and copy all the test images into it   
Then Run
> python test_pretrained.py


### To Test model that was trained from scratch

Create a directory named **test_loader** and copy all the test images into it   
Then Run
> python test_scratch.py


### To train model using pretrained network

Run the below command
> python train_pretrained.py

### To train model from scratch

Run the below command
> python train_scratch.py


### Important Files and Folders

1. model.py - Implementation of Model Architecture
2. logs - Contains carried out experiments
3. helper.py - Functions needed for training and testing
4. part_3_pretrained_mnist_model_tb ( Downloaded from drive) - Contains Model Weights for the model trained on top of pretrained model
5. part_3_scratch_mnist_model_tb( Downloaded from drive) - Contains Model Weights for the models trained from scratch 
6. pretrained_custom_mnist.pth.tar ( Downloaded from drive) - Contains Model Weights for the model used as pretrained network in Part-2

### Report

**part3_report.pdf** briefs about the dataset,experiment details, observations and results

