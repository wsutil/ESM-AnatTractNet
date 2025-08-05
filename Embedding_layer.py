import torch.nn as nn

class ROIFeatureExtractor(nn.Module):
    def __init__(self, embedding_layer, ROI_EMBEDDING_DIM, hidden_dim):
        super(ROIFeatureExtractor, self).__init__()
        self.embedding = embedding_layer
        self.mlp = nn.Sequential(
            nn.Linear(ROI_EMBEDDING_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ROI_EMBEDDING_DIM)
        )
        print(f"using ROI with emb: {ROI_EMBEDDING_DIM}")

    def forward(self, roi_input):
        embedded = self.embedding(roi_input)
        features = self.mlp(embedded)  
        return features
