from torch.utils.data import Dataset

from PIL import Image
import torchvision.transforms as transforms

class CausalImageDataset(Dataset):
    def __init__(self,image_paths, confounds, treatments = None, outcomes = None,transform = None):
        self.image_paths = image_paths
        self.confounds = confounds
        self.treatments = treatments
        self.outcomes = outcomes

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406] , std = [0.229, 0.224, 0.225])
                ]
            )
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self,idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        confounds = self.confounds[idx]
        treatment = self.treatments[idx] if self.treatments is not None else -1
        outcome = self.outcomes[idx] if self.outcomes is not None else -1
        return image , confounds, treatment, outcome

    