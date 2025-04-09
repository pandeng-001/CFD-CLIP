import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# CLIP + Classifier validation accuracy
class ComplexClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=100):
        super(ComplexClassifier, self).__init__()
        
        # First fully-connected layer + ReLU activation
        self.fc1 = nn.Linear(input_dim, 2048)
        self.bn1 = nn.BatchNorm1d(2048)  # BatchNorm 
        self.relu1 = nn.ReLU()
        
        # Second fully-connected layer + ReLU activation
        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)  # BatchNorm 
        self.relu2 = nn.ReLU()

        # Dropout layer (for overfitting prevention)
        self.dropout = nn.Dropout(0.5)
        
        # Third fully-connected layer + ReLU activation
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)  # BatchNorm 
        self.relu3 = nn.ReLU()
        
        # output
        self.fc4 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.relu1(self.bn1(self.fc1(x)))  
        x = self.relu2(self.bn2(self.fc2(x)))  
        x = self.dropout(x)                    
        x = self.relu3(self.bn3(self.fc3(x)))  
        x = self.fc4(x)                        
        return x
        # return self.dropout(x)
    

# Contrastive loss function
def contrastive_loss(A, B, temperature=0.07):

    # Compute cosine similarity between A and B
    A_norm = F.normalize(A, p=2, dim=-1)
    B_norm = F.normalize(B, p=2, dim=-1)
    
    # Calculate similarity matrix (pairwise cosine similarities between A and B)
    similarity_matrix = torch.matmul(A_norm, B_norm.T)
    
    # Labels are diagonal elements (matching positions in A and B)
    labels = torch.arange(A.size(0), device=A.device)
    
    # Compute contrastive loss using cross-entropy
    loss = F.cross_entropy(similarity_matrix / temperature, labels)
    
    return loss


#   model + classifier head
class combined_model_classifier(nn.Module):
    def __init__(self, model1, model2):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
    def forward(self, x):
        x = self.model1(x)
        x = self.model2(x)
        return x


class FeatureAlign(nn.Module):
    # input_dim 960 576
    def __init__(self, input_dim=576, output_dim=512):
        super(FeatureAlign, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Global Average Pooling (GAP)
        batch, channels, height, width = x.size()
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(batch, channels)  # [batch, 576]
        x = self.projection(x)  # [batch, 512]
        return x


if __name__ == "__main__":
    A =torch.randn((2,56))
    fa = FeatureAlign(input_dim=256)
    X = fa(A)
    print(X)
