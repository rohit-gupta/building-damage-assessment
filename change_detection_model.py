import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation, use_bn=True, padding_type=None, apply_relu=True):
        super(ConvLayer, self).__init__()
        self.apply_rely = apply_relu
        self.padding_type = padding_type
        if padding_type == "replication":
            self.pad = nn.ReplicationPad2d((kernel_size * dilation - dilation + 1) // 2)  # SAME padding
        elif padding_type == "reflection":
            self.pad = nn.ReflectionPad2d((kernel_size * dilation - dilation + 1) // 2)   # SAME padding
        self.conv = nn.Conv2d(int(in_channels), int(out_channels),
                              int(kernel_size),
                              dilation=int(dilation),
                              bias=False)
        # if kernel_size == 1:
        #     nn.init.ones_
        nn.init.kaiming_uniform_(self.conv.weight)
        print(self.conv.weight.data.shape)
        # nn.init.zeros_(self.conv.weight)
        # nn.init.kaiming_uniform_(self.conv.weight)
        if use_bn:
            self.bn = nn.BatchNorm2d(int(out_channels), eps=0.001)
            nn.init.ones_(self.bn.weight)
            nn.init.zeros_(self.bn.bias)
        else:
            self.bn = None

        # nn.init.zeros_(self.conv.weight) # Not using bias for now


    def forward(self, x):
        if self.padding_type: # 1x1 Convs don't need padding
            x = self.pad(x)
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.apply_rely:
            return F.relu(x, inplace=True)
        else:
            return x


class MultiScaleContextLayer(nn.Module):

    def __init__(self, in_channels, feature_channels, out_channels, kernel_scales=None, dilation_scales=None,
                 use_bn=True, padding_type="replication", is_final_layer=False):
        super(MultiScaleContextLayer, self).__init__()
        assert len(kernel_scales) == len(dilation_scales), "branch scales are mismatched"
        num_branches = len(kernel_scales)
        assert feature_channels % num_branches == 0, "number of feature channels not divisible by number of branches"

        if kernel_scales[0] > 1:
            local_branch_padding_type = padding_type
        else:
            local_branch_padding_type = None

        self.local_branch = ConvLayer(in_channels, feature_channels / num_branches,
                                      kernel_size=kernel_scales[0],
                                      dilation=dilation_scales[0],
                                      use_bn=use_bn, padding_type=local_branch_padding_type)

        if num_branches >= 2:
            self.non_local_branch1 = ConvLayer(in_channels, feature_channels / num_branches,
                                               kernel_size=kernel_scales[1],
                                               dilation=dilation_scales[1],
                                               use_bn=use_bn, padding_type=padding_type)
        else:
            self.non_local_branch1 = None
        if num_branches == 3:
            self.non_local_branch2 = ConvLayer(in_channels, feature_channels / num_branches,
                                               kernel_size=kernel_scales[2],
                                               dilation=dilation_scales[2],
                                               use_bn=use_bn, padding_type=padding_type)

        else:
            self.non_local_branch2 = None

        if is_final_layer:
            self.merge_layer = None
        else:
            self.merge_layer = ConvLayer(feature_channels, out_channels,
                                         kernel_size=1,
                                         dilation=1,
                                         use_bn=False)

    def forward(self, x):
        local_features = self.local_branch(x)

        features = [local_features]

        if self.non_local_branch1:
            non_local_features1 = self.non_local_branch1(x)
            features += [non_local_features1]

        if self.non_local_branch2:
            non_local_features2 = self.non_local_branch2(x)
            features += [non_local_features2]

        merged_features = torch.cat(features, dim=1)
        if self.merge_layer:
            output = self.merge_layer(merged_features)
        else:
            output = merged_features

        return output


class ChangeDetectionNet(nn.Module):
    def __init__(self, classes, num_layers=1, feature_channels=None, kernel_scales=None, dilation_scales=None,
                 use_bn=True, padding_type="replication"):
        super(ChangeDetectionNet, self).__init__()
        if kernel_scales is None:
            kernel_scales = [3, 13, 23]
        if dilation_scales is None:
            dilation_scales = [2, 4, 8] # Receptive fields = 3 + 1*(3-1) == 5, 13 + 3*(13-1) == 49, 23 + 7*(23-1) == 177
        if feature_channels is None:
            feature_channels = len(kernel_scales) * classes

        self.first_layer = MultiScaleContextLayer(in_channels=2*classes,
                                                  feature_channels=feature_channels,
                                                  out_channels=classes,
                                                  kernel_scales=kernel_scales,
                                                  dilation_scales=dilation_scales,
                                                  use_bn=use_bn, padding_type=padding_type,
                                                  is_final_layer=(num_layers == 1))

        higher_layers = []
        for layer_num in range(num_layers-1):
            higher_layers.append(MultiScaleContextLayer(in_channels=3*classes, # Segmentation Outputs + Previous layer Detected Change
                                                        feature_channels=feature_channels,
                                                        out_channels=classes,
                                                        kernel_scales=kernel_scales,
                                                        dilation_scales=dilation_scales,
                                                        use_bn=use_bn, padding_type=padding_type,
                                                        is_final_layer=(num_layers == layer_num-2)))

        self.higher_layers = nn.ModuleList(higher_layers)

        self.final_layer = ConvLayer(classes, classes,
                                     kernel_size=1,
                                     dilation=1,
                                     use_bn=False, apply_relu=False)

    def forward(self, x):
        first_pred = self.first_layer(x)
        preds = [first_pred]
        for layer in self.higher_layers:
            preds.append(layer(torch.cat((x, preds[-1]), dim=1))) # Segmentation outputs + Prev Layer Change Prediction

        return self.final_layer(preds[-1])


class RegressChangeNet(nn.Module):
    def __init__(self):
        super(RegressChangeNet, self).__init__()
        self.conv = nn.Conv2d(in_channels=10, out_channels=5,
                              kernel_size=1, stride=1, padding=0, dilation=1,
                              groups=1, bias=False, padding_mode='zeros')

        nn.init.kaiming_uniform_(self.conv.weight)

    def forward(self, x):
        return self.conv(x)
