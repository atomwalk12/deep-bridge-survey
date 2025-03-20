import torch
import torch.nn as nn
import json
import os

class ToyNetwork(nn.Module):
    def __init__(self, config_path='network_config.json'):
        super(ToyNetwork, self).__init__()
        
        # Read configuration from JSON file
        self.config = self._load_config(config_path)
        
        # Network parameters from config
        network_config = self.config['network']
        self.batch_size = network_config['batch_size']
        self.num_classes = network_config['num_classes']
        self.in_channels = network_config['in_channels']
        self.input_height = network_config['input_height']
        self.input_width = network_config['input_width']
        
        # Build convolutional layers from config
        conv_layers = []
        in_channels = self.in_channels
        for conv_config in self.config['conv_layers']:
            conv_layers.append(
                nn.Conv2d(
                    in_channels,
                    conv_config['out_channels'],
                    kernel_size=conv_config['kernel_size'],
                    stride=conv_config['stride'],
                    padding=conv_config['padding']
                )
            )
            conv_layers.append(nn.ReLU(inplace=True))
            in_channels = conv_config['out_channels']
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate the flattened size after convolutions
        flattened_size = in_channels * self.input_height * self.input_width
        
        # Build fully connected layers from config
        fc_layers = []
        in_features = flattened_size
        for fc_config in self.config['fc_layers']:
            out_features = fc_config['out_features']
            # Handle special case for output layer
            if out_features == -1:
                out_features = self.num_classes
            
            fc_layers.append(nn.Linear(in_features, out_features))
            # Add ReLU after all but the last layer
            if out_features != self.num_classes:
                fc_layers.append(nn.ReLU(inplace=True))
            
            in_features = out_features
        
        self.fc_layers = nn.Sequential(*fc_layers)
    
    def _load_config(self, config_path):
        """Load network configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Error loading configuration file: {e}")
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x
