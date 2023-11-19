from utils.data_helper import create_dataset_and_dataloader
import argparse
from models.lstm_models import LSTMModel
from models.transformer_models import TransformerModel
from models.cnn_models import CNNModel

from config.general_cfg import (LSTM_INPUT_SIZE, 
                                LSTM_HIDDEN_SIZE, 
                                LSTM_NUM_LAYERS,
                                LSTM_NUM_EPOCHS,
                                LSTM_LEARNING_RATE,
                                NUM_CLASSES,
                                TRANSFORMER_NHEAD,
                                TRANSFORMER_D_MODEL,
                                TRANSFORMER_NUM_LAYERS,
                                TRANSFORMER_LEARNING_RATE,
                                TRANSFORMER_NUM_EPOCHS,
                                CNN_D_MODEL,
                                CNN_LEARNING_RATE,
                                CNN_NUM_EPOCHS)
import torch.nn as nn
import torch.optim as optim
import models.trainer as trainer 

def main():

    train_dataloader, eval_dataloader, test_dataloader = create_dataset_and_dataloader()
    dataloader = (train_dataloader, eval_dataloader, test_dataloader )
    # Define the command line arguments
    parser = argparse.ArgumentParser(description='Choose a model for training.')
    parser.add_argument('--model', type=str, choices=['lstm','transformer' ,'cnn', 'dnn'], required=True, help='The model to use for training.')

    # Parse the command line arguments
    args = parser.parse_args()

    train_elems ={}
    train_elems['criterion'] = nn.CrossEntropyLoss()
    # Depending on the chosen model, initialize the appropriate model
    if args.model == 'lstm':
        train_elems['model'] = LSTMModel(LSTM_INPUT_SIZE,
                                         LSTM_HIDDEN_SIZE,
                                         LSTM_NUM_LAYERS,
                                         NUM_CLASSES)
        
        train_elems['optimizer'] = optim.Adam(train_elems['model'].parameters(), 
                                              lr=LSTM_LEARNING_RATE)
        train_elems['num_epochs'] = LSTM_NUM_EPOCHS


    elif args.model == 'transformer':
        train_elems['model'] = TransformerModel(d_model=TRANSFORMER_D_MODEL,
                                                nhead=TRANSFORMER_NHEAD,
                                                num_layers=TRANSFORMER_NUM_LAYERS,
                                                num_classes=NUM_CLASSES)
        
        train_elems['optimizer'] = optim.Adam(train_elems['model'].parameters(), 
                                              lr=TRANSFORMER_LEARNING_RATE)
        train_elems['num_epochs'] = TRANSFORMER_NUM_EPOCHS

    elif args.model == 'cnn':
        train_elems['model'] = CNNModel(CNN_D_MODEL,NUM_CLASSES)
        train_elems['optimizer'] = optim.Adam(train_elems['model'].parameters(), 
                                        lr=CNN_LEARNING_RATE)
        train_elems['num_epochs'] = CNN_NUM_EPOCHS
    

    trainer.train(dataloader=dataloader,train_elems=train_elems)    


if __name__=='__main__':
    main()