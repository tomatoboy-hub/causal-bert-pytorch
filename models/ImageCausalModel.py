import timm
import torch
from torch import nn

CUDA = (torch.cuda.device_count() > 0)

def make_confound_vector(ids, vocab_size, use_counts = False):
    vec = torch.zeros(ids.shape[0],vocab_size)
    ones = torch.ones_like(ids,dtype = torch.float)
    
    if CUDA:
        vec = vec.cuda()
        ones = ones.cuda()
        ids = ids.cuda()
    vec[:,1] = 0.0
    if not use_counts:
        vec = (vec != 0).float()
    return vec.float()


class ImageCausalModel(nn.Module):
    """The model itself."""
    def __init__(self, num_labels = 2,pretrained_model_names = "resnet50"):
        super().__init__()

        self.num_labels = num_labels

        self.base_model = timm.create_model(pretrained_model_names,pretrained = True)
        self.base_model.fc = nn.Identity()

        # 因果推論用の追加レイヤー
        self.classifier = nn.Linear(self.base_model.num_features, num_labels)
        self.Q_cls = nn.ModuleDict()

        # self.base_model.num_features は、事前学習済みの画像モデルからの特徴量サイズです。
        input_size = self.base_model.num_features + self.num_labels

        for T in range(2):
            # ModuleDict keys have to be strings..
            self.Q_cls['%d' % T] = nn.Sequential(
                nn.Linear(input_size, 200),
                nn.ReLU(),
                nn.Linear(200, self.num_labels))
        

        self.g_cls = nn.Linear(self.base_model.num_features + self.num_labels, 
            self.num_labels)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self,images, confounds, treatment=None, outcome = None):
        features = self.base_model(images)
        # print("features")
        # print("C", confounds.shape)
        # print(confounds)
        # print(confounds.unsqueeze(1).shape)
        C = make_confound_vector(confounds.unsqueeze(1), self.num_labels)
        # print("C",C.shape) 
        inputs = torch.cat((features, C), dim =  1)
        g_logits = self.g_cls(inputs)
        g_prob = torch.sigmoid(g_logits)

        Q_logits_T0 = self.Q_cls['0'](inputs)
        Q_logits_T1 = self.Q_cls['1'](inputs)

        Q_prob_T0 = torch.sigmoid(Q_logits_T0)
        Q_prob_T1 = torch.sigmoid(Q_logits_T1)
        if outcome is not None:
            return g_prob, Q_prob_T0, Q_prob_T1, g_logits, Q_logits_T0, Q_logits_T1
        else:
            return g_prob, Q_prob_T0, Q_prob_T1,
        