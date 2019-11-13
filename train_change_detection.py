
import torch
from change_detection_model import ChangeDetectionNet


changenet = ChangeDetectionNet(classes=5, num_layers=2, feature_channels=15,
                               kernel_scales=[3, 13, 23], dilation_scales=[2, 4, 8],
                               use_bn=True, padding_type="replication")
changenet = changenet.cuda()
x = torch.rand(8,10,1024,1024)
y = changenet(x.cuda())