# Part-2

## Note

1. Before proceeding go to this link: [This link](https://drive.google.com/drive/folders/13toHSi0s7wqKRJu7VKACIP-0vLEdwLvc?usp=sharing)
2. Then download the file named **important_part2.zip**
3. Extract all files and folders in this directory

### Run all experiments
 Run **task2_2_final.ipynb** this to reproduce all results

### To Test model that was trained using pretrained network

Create a directory named **test_loader** and copy all the test images into it   
Then Run
> python test_pretrained_model.py


### To Test model that was trained from scratch

Create a directory named **test_loader** and copy all the test images into it   
Then Run
> python test_scratch_model.py


### To train model using pretrained network

Run the below command
> python train_standard_mnist_pretrained.py

### To train model from scratch

Run the below command
> python train_standard_mnist_scratch.py


### To train model on Custom MNIST Dataset 

Run the below command
> python train_standard_mnist_scratch.py

### Important Files and Folders

1. model.py - Implementation of Model Architecture
2. logs( Downloaded from drive)  - Contains carried out experiments
3. helper.py - Functions needed for training and testing
4. part_2_custom_mnist_model_tb ( Downloaded from drive) - Contains Model Weights for the model trained on Custom MNIST Dataset
5. part_2_standard_mnist( Downloaded from drive) - Contains Model Weights for both the models trained from scratch and trained on top of pretrained model

### Report

**part2_report.pdf** briefs about the dataset,experiment details, observations and results
