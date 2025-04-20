import torch
import torch.nn as nn
import torchvision.models as models


'''class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features'''
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        for param in resnet.parameters():
            param.requires_grad_(False)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
    
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.bn(self.embed(features))
        return features
    

'''class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(num_embeddings = vocab_size,
                                  embedding_dim = embed_size)
        
        self.lstm = nn.LSTM(input_size = embed_size,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            batch_first = True) 
        
        self.linear = nn.Linear(in_features = hidden_size,
                                out_features = vocab_size)
        
        
        
    
    def forward(self, features, captions):
        captions = captions[:, :-1]
        embedding = self.embed(captions)
        embedding = torch.cat((features.unsqueeze(dim = 1), embedding), dim = 1)
        lstm_out, hidden = self.lstm(embedding)
        outputs = self.linear(lstm_out)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        predicted_sentence = []
        for index in range(max_len):
            
            
            lstm_out, states = self.lstm(inputs, states)

            
            lstm_out = lstm_out.squeeze(1)
            outputs = self.linear(lstm_out)
            
            
            target = outputs.max(1)[1]
            
            
            predicted_sentence.append(target.item())
            
            
            inputs = self.embed(target).unsqueeze(1)
            
        return predicted_sentence'''

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.num_layers = num_layers  # Store num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)
    
    def forward(self, features, captions):
        captions = captions[:, :-1]
        embedding = self.embed(captions)
        embedding = torch.cat((features.unsqueeze(dim=1), embedding), dim=1)
        lstm_out, hidden = self.lstm(embedding)
        outputs = self.linear(lstm_out)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        """Accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len)"""
        predicted_sentence = []
        batch_size = inputs.size(0)  # Should be 1 for app.py
        
        # Initialize hidden states if None
        if states is None:
            h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(inputs.device)
            c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(inputs.device)
            states = (h, c)
        
        inputs = inputs.unsqueeze(1)  # [batch_size, 1, embed_size]
        for _ in range(max_len):
            lstm_out, states = self.lstm(inputs, states)  # lstm_out: [batch_size, 1, hidden_size]
            lstm_out = lstm_out.squeeze(1)  # [batch_size, hidden_size]
            outputs = self.linear(lstm_out)  # [batch_size, vocab_size]
            _, target = outputs.max(1)  # [batch_size]
            predicted_sentence.append(target.item())  # Assuming batch_size=1
            inputs = self.embed(target).unsqueeze(1)  # [batch_size, 1, embed_size]
        
        return predicted_sentence