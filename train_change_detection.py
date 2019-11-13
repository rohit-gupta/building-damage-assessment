
import torch
from change_detection_model import ChangeDetectionNet



changenet = ChangeDetectionNet(classes=5, num_layers=3, feature_channels=15,
                               kernel_scales=[3, 11, 19], dilation_scales=[2, 4, 8],
                               use_bn=True, padding_type="replication")


print(changenet)
for p in changenet.parameters():
    print(p.numel())
cuda = torch.device('cuda')
changenet = changenet.cuda()
x = torch.rand(8,10,1024,1024)
x = x.cuda()
y = changenet(x)

print(x.shape, len(y), y[0].shape)
