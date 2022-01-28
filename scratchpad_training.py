# flake8: noqa

import gc
from collections import OrderedDict
from torch import optim
from torch import nn
from ntm_classifier.model_training import CustomDataset, ready_dataframe
import torch
from tqdm import tqdm

images_dataset = CustomDataset(ready_dataframe())
dl = torch.utils.data.DataLoader(images_dataset, batch_size=5)



resnet50 = torch.hub.load(
    'NVIDIA/DeepLearningExamples:torchhub',
    # num_classes=13,
    'nvidia_resnet50',
    pretrained=True,
)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
resnet_50_14 = nn.Sequential(OrderedDict({
    'resnet': resnet50,
    'fc': nn.Linear(1000, 14),
    'output': nn.LogSoftmax(dim=1)
})).to(device)

sample_shape = torch.rand(size=(5, 3, 224, 224)).to(device)
traced = torch.jit.tract(resnet_50_14, sample_shape)
traced.get_submodule('resnet')
torch.jit.save(traced.get_submodule('resnet'), resource_filename('ntm_data', 'base_resnet50.pt'))


# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(resnet_50_14.parameters(), lr=0.005, momentum=0.9)


# epochs = 5
# scheduler = torch.optim.lr_scheduler.OneCycleLR(
#     optimizer, max_lr=0.01, steps_per_epoch=len(dl), epochs=epochs)


# def train_epoch(
#         network=resnet_50_14,
#         data=dl):

#     train_len = len(data)
#     running_loss = 0.0
#     loop = tqdm(enumerate(data), total=train_len)

#     loop.desc = ('train batch: 0| Cross Entropy: 0.0')

#     for i, data in loop:
#         inputs, labels = data

#         if inputs.shape[0] == 1:
#             continue

#         optimizer.zero_grad()

#         outputs = network(inputs.to(device))
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
#         # scheduler.print_lr()
#         running_loss += loss.item()
#         # if i % (train_len//4) == 0:
#         loop.desc = ('train batch: %2d| Cross Entropy: %.3f' %
#                      (i, running_loss / (i + 1)))

#     return running_loss / min(1, train_len)

# for i in range(1, epochs):
#     training_loss = str(train_epoch())
#     print(f"Epoch {i}| Training Loss: {training_loss}")
#     # testing_loss = str(test_epoch())
#     # print(f"Epoch {i}| Testing Loss: {testing_loss}")

#     gc.collect()
#     torch.cuda.empty_cache()

# torch.save(resnet_50_14, f'./resnet_50_13_{epochs}.pt')


# from ntm_classifier.load_resources import (
#     load_classification_table,
#     load_report_image,
#     process_mappings_group,
# )
# from ntm_classifier.preprocess import preprocess
# primary_mappings = process_mappings_group('primary')


# df = load_classification_table()

# def get_estimate(i):
#     guess = resnet_50_14(preprocess(load_report_image(df['file'].loc[i])).unsqueeze(0).to(device)).argmax()
#     return df['tags'].loc[i], guess.item(), primary_mappings.get(guess.item())


# from random import randint
# i = randint(0, 1000)
# get_estimate(i), i


# from ntm_classifier.report import primary_report, get_heatmap
# # labels = list(primary_mappings.keys())
# labels = list((p.lower() for p in primary_mappings.values()))

# matrix = primary_report(df, labels=labels, model=resnet_50_14)
# heat = get_heatmap(matrix, labels)

# primary_mappings.values()


# import torch
# from ntm_classifier.load_resources import process_mappings_group, load_classification_table
# from ntm_classifier.report import primary_report, get_heatmap
# df = load_classification_table()

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# resnet50 = torch.hub.load(
#     'NVIDIA/DeepLearningExamples:torchhub',
#     # num_classes=13,
#     'nvidia_resnet50',
#     pretrained=True,
# )

# primary_mappings = process_mappings_group('primary')
# labels = list((p.lower() for p in primary_mappings.values()))
# resnet = torch.load('./resnet_50_13_10.pt')
# matrix = primary_report(df, labels=labels, model=other_resnet)
# heat = get_heatmap(matrix, labels)


# other_resnet = torch.load('../ntm_data/resnet_50_13_5.pt')
# matrix2 = primary_report(df, labels=labels, model=other_resnet)
# heat = get_heatmap(matrix2, labels)

from ntm_classifier.training.image_dataset import ImageDataset
from ntm_classifier.training.trainer import Trainer
secs = ImageDataset('secondary_tags.csv', 'secondary', 'secondary')
secs[0:2]
secs[0]


trainer = Trainer(secs)

trainer.train()