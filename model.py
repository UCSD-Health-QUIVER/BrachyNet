import torch
import torch.nn as nn
import torch.nn.functional as F



class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation =1):
        super(SingleConv, self).__init__()

        self.single_conv = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, groups=in_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=True, dilation =1), # depthwise
            nn.InstanceNorm3d(in_ch, affine=True),
            #nn.BatchNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch, out_ch, kernel_size=1, padding=0, stride=1, bias=True), # pointwise
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True)

        )

    def forward(self, x):
        return self.single_conv(x)


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()


        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, groups=in_ch, kernel_size=3, padding=1, stride=1, bias=True, dilation =1), # depthwise
            nn.InstanceNorm3d(in_ch, affine=True),
            #nn.BatchNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch, out_ch, kernel_size=1, padding=0, stride=1, bias=True), # pointwise
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True)

        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_ch, list_ch, dropout_prob):
        super(Encoder, self).__init__()
        self.encoder_1 = nn.Sequential(
            SingleConv(in_ch, list_ch[1], kernel_size=3, stride=1, padding='same', dilation = 3),
            SingleConv(list_ch[1], list_ch[1], kernel_size=3, stride=1, padding='same', dilation = 2),
            SingleConv(list_ch[1], list_ch[1], kernel_size=3, stride=1, padding=1),
            torch.nn.Dropout3d(p=dropout_prob)
        )
        self.encoder_2 = nn.Sequential(
            SingleConv(list_ch[1], list_ch[2], kernel_size=3, stride=2, padding=1),
            SingleConv(list_ch[2], list_ch[2], kernel_size=3, stride=1, padding="same", dilation = 2),
            torch.nn.Dropout3d(p=dropout_prob)
        )
        self.encoder_3 = nn.Sequential(
            SingleConv(list_ch[2], list_ch[3], kernel_size=3, stride=2, padding=1),
            SingleConv(list_ch[3], list_ch[3], kernel_size=3, stride=1, padding='same', dilation = 2),
            torch.nn.Dropout3d(p=dropout_prob)
        )
        self.encoder_4 = nn.Sequential(
            SingleConv(list_ch[3], list_ch[4], kernel_size=3, stride=2, padding=1),
            SingleConv(list_ch[4], list_ch[4], kernel_size=3, stride=1, padding=1),
            torch.nn.Dropout3d(p=dropout_prob)
        )
        self.encoder_5 = nn.Sequential(
            SingleConv(list_ch[4], list_ch[5], kernel_size=3, stride=2, padding=1),
            SingleConv(list_ch[5], list_ch[5], kernel_size=3, stride=1, padding=1),
            torch.nn.Dropout3d(p=dropout_prob)
        )

    def forward(self, x):
        out_encoder_1 = self.encoder_1(x)
        out_encoder_2 = self.encoder_2(out_encoder_1)
        out_encoder_3 = self.encoder_3(out_encoder_2)
        out_encoder_4 = self.encoder_4(out_encoder_3)
        out_encoder_5 = self.encoder_5(out_encoder_4)

        return [out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4, out_encoder_5]



class Decoder(nn.Module):
    def __init__(self, list_ch, dropout_prob):
        super(Decoder, self).__init__()

        self.upconv_4 = UpConv(list_ch[5], list_ch[4])
        self.decoder_conv_4 = nn.Sequential(
            SingleConv(2 * list_ch[4], list_ch[4], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[4], list_ch[4], kernel_size=3, stride=1, padding=1),
            torch.nn.Dropout3d(p=dropout_prob)
        )
        self.upconv_3 = UpConv(list_ch[4], list_ch[3])
        self.decoder_conv_3 = nn.Sequential(
            SingleConv(2 * list_ch[3], list_ch[3], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[3], list_ch[3], kernel_size=3, stride=1, padding=1),
            torch.nn.Dropout3d(p=dropout_prob)
        )
        self.upconv_2 = UpConv(list_ch[3], list_ch[2])
        self.decoder_conv_2 = nn.Sequential(
            SingleConv(2 * list_ch[2], list_ch[2], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[2], list_ch[2], kernel_size=3, stride=1, padding=1),
            torch.nn.Dropout3d(p=dropout_prob)
        )
        self.upconv_1 = UpConv(list_ch[2], list_ch[1])
        self.decoder_conv_1 = nn.Sequential(
            SingleConv(2 * list_ch[1], list_ch[1], kernel_size=3, stride=1, padding=1)
        )

    def forward(self, out_encoder):
        out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4, out_encoder_5 = out_encoder

        out_decoder_4 = self.decoder_conv_4(
            torch.cat((self.upconv_4(out_encoder_5), out_encoder_4), dim=1)
        )
        out_decoder_3 = self.decoder_conv_3(
            torch.cat((self.upconv_3(out_decoder_4), out_encoder_3), dim=1)
        )
        out_decoder_2 = self.decoder_conv_2(
            torch.cat((self.upconv_2(out_decoder_3), out_encoder_2), dim=1)
        )
        out_decoder_1 = self.decoder_conv_1(
            torch.cat((self.upconv_1(out_decoder_2), out_encoder_1), dim=1)
        )

        return out_decoder_1


class BaseUNet(nn.Module):
    def __init__(self, in_ch, list_ch, dropout_prob):
        super(BaseUNet, self).__init__()
        self.encoder = Encoder(in_ch, list_ch, dropout_prob)
        self.decoder = Decoder(list_ch, dropout_prob)

        # init
        self.initialize()

    @staticmethod
    def init_conv_IN(modules):
        for m in modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.InstanceNorm3d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def initialize(self):
        print('# random init encoder weight using nn.init.kaiming_uniform !')
        self.init_conv_IN(self.decoder.modules)
        print('# random init decoder weight using nn.init.kaiming_uniform !')
        self.init_conv_IN(self.encoder.modules)

    def forward(self, x):
        out_encoder = self.encoder(x)
        out_decoder = self.decoder(out_encoder)


        return out_decoder


class Model(nn.Module):
    def __init__(self, in_ch, out_ch, list_ch_A, list_ch_B,dropout_prob =0.0):

        super(Model, self).__init__()

        # list_ch records the number of channels in each stage, eg. [-1, 32, 64, 128, 256, 512]
        self.net_A = BaseUNet(in_ch, list_ch_A, dropout_prob)
        self.net_B = BaseUNet(in_ch + list_ch_A[1], list_ch_B, dropout_prob)

        # final convolutions to smooth the output
        self.conv_out_A = nn.Sequential(
                        SingleConv(list_ch_A[1], list_ch_A[1]//2, kernel_size=3, stride=1, padding=1),
                        SingleConv(list_ch_A[1]//2, list_ch_A[1]//4, kernel_size=3, stride=1, padding=1),
                        SingleConv(list_ch_A[1]//4, out_ch, kernel_size=1, stride=1, padding=0),
                        nn.ReLU()
                        )
        self.conv_out_B = nn.Sequential(
                        SingleConv(list_ch_B[1], list_ch_B[1]//2, kernel_size=3, stride=1, padding=1),
                        SingleConv(list_ch_B[1]//2, list_ch_B[1]//4, kernel_size=3, stride=1, padding=1),
                        SingleConv(list_ch_B[1]//4, out_ch, kernel_size=1, stride=1, padding=0),
                        nn.ReLU()
                        )
        self.final_relu = nn.ReLU()




    def forward(self, x):
        out_net_A = self.net_A(x) 

        output_A = self.conv_out_A(out_net_A)
        output_A = self.final_relu(output_A)

        out_net_B = self.net_B(torch.cat((out_net_A, x), dim=1))

        output_B = self.conv_out_B(out_net_B)
        output_B = self.final_relu(output_B)
        
        return [output_A, output_B]

