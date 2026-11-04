import torch
from torch.utils.data import DataLoader, random_split,Subset
from torchvision.datasets import *
from torchvision.datasets import *
from torchvision.transforms.v2 import *
from torchvision.transforms.v2.functional import *

seed = 9970
torch.manual_seed(seed)
h,w=128,128
transform = Compose(
        [
            Resize((h, w)),
            RandomHorizontalFlip(),
            ToImage(),
            ToDtype(torch.float, scale=True),
        ]
    )

dataset_dir = "/data1/dataset/jssnu005/sys/lh/FaceScrub"
ds = ImageFolder(root=dataset_dir, transform=transform)
ds_len = len(ds)
test_size = ds_len // 10
train_size = ds_len - test_size
half = train_size // 2
shuffled_indices = torch.randperm(
    ds_len, generator=torch.Generator().manual_seed(seed)
).tolist()
idx_part1 = shuffled_indices[:half]
idx_part2 = shuffled_indices[half :train_size]
idx_test = shuffled_indices[train_size:]
part1_ds=Subset(ds, idx_part1)
part2_ds=Subset(ds, idx_part2)
test_ds=Subset(ds, idx_test)
#class_num = len(ds.classes)
#train_ds_len = len(train_ds)
#test_ds_len = len(test_ds)
