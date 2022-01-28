# flake8: noqa

# import gc
# from collections import OrderedDict
# from torch import optim
# from torch import nn
# from ntm_classifier.model_training import CustomDataset, ready_dataframe
# import torch
# from tqdm import tqdm

# images_dataset = CustomDataset(ready_dataframe())
# dl = torch.utils.data.DataLoader(images_dataset, batch_size=5)



# resnet50 = torch.hub.load(
#     'NVIDIA/DeepLearningExamples:torchhub',
#     # num_classes=13,
#     'nvidia_resnet50',
#     pretrained=True,
# )


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# resnet_50_14 = nn.Sequential(OrderedDict({
#     'resnet': resnet50,
#     'fc': nn.Linear(1000, 14),
#     'output': nn.LogSoftmax(dim=1)
# })).to(device)

# sample_shape = torch.rand(size=(5, 3, 224, 224)).to(device)
# traced = torch.jit.tract(resnet_50_14, sample_shape)
# traced.get_submodule('resnet')
# torch.jit.save(traced.get_submodule('resnet'), resource_filename('ntm_data', 'base_resnet50.pt'))

from ntm_classifier.training.image_dataset import ImageDataset
from ntm_classifier.training.trainer import Trainer
secs = ImageDataset('secondary_tags.csv', 'secondary', 'secondary')
secs[0:2]
secs[0]
secs.na_value+1

trainer = Trainer(secs, n_labels=20)

trainer.train()