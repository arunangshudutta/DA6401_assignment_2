# DA6401_assignment_2

## File organization
Assignment_2_QA2.ipynb contains the code for part A question 2: runnnig a wandb sweep on models created from scratch.   
Assignment_2_QA4.ipynb contains the code for part A question 4: Training and evaluating the best model from the sweep. 
Assignment_2_QB3.ipynb contains the code for part B question 3: Fine-tuning the pre-trained ResNet50 

## Importent functions

- To create a CNN from scratch we can use "FlexibleCNN()". Example code

model = FlexibleCNN(
    in_channels=3,
    conv_channels=[32, 32, 32, 32, 32],
    kernel_sizes=[3, 3, 3, 3, 3],
    dense_size = 256,
    activation_fn=nn.ReLU(),
    dropout_prob=0.4,
    num_classes=10,
    input_size=224
).to(device)

this lets us choose the number of convolution layers, filter shape, number of filters, dropout probability etc.

- To split dataset into training ad validation sets with specific trasforms, use the function "load_split_dataset(dataset_path, train_ratio = 0.8, data_aug = False ,batch_size=64)".
- To train the model use "train(model, loader, optimizer, criterion, device)", this returns the trining loss and accuracy.
- To evaluare the model use "evaluate(model, loader, criterion, device)", this returns accuracy and loss any dataset in the dataloader passed as input.
