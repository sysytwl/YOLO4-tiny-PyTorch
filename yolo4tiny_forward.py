import torch
import numpy as np
import torch.nn.functional as F

def FusedBasicConv(x,w,b,s,p=1):
    x=F.conv2d(input=x,weight=w,bias=b,stride=s,padding=p)
    return F.leaky_relu(input=x,negative_slope=0.1)

def FusedResBlock(x,n,c,w1,w2,w3,w4,b1,b2,b3,b4):
    x=F.conv2d(input=x,weight=w1,bias=b1,stride=1,padding=1)
    x=F.leaky_relu(x,negative_slope=0.1)
    route=x
    #
    x=F.conv2d(input=x[:,n//2:n,:,:],weight=w2,bias=b2,stride=1,padding=1)
    route1=F.leaky_relu(input=x,negative_slope=0.1)
    #
    x=F.conv2d(input=route1,weight=w3,bias=b3,stride=1,padding=1)
    x=F.leaky_relu(input=x,negative_slope=0.1)
    #
    x=torch.cat([x,route1],dim=1)
    #
    x=F.conv2d(input=x,weight=w4,bias=b4,stride=1,padding=0)
    feat=F.leaky_relu(input=x,negative_slope=0.1)
    #
    x=torch.cat([route,feat],dim=1)
    x=F.max_pool2d(x,kernel_size=2,stride=2)
    return x,feat

class MyCSPdarknet53_tiny:
    def __init__(self):
        self.basic_conv1_w=None
        self.basic_conv1_b=None
        self.basic_conv2_w=None
        self.basic_conv2_b=None
        self.basic_conv3_w=None
        self.basic_conv3_b=None

    def load_weight(self,dir):
        #basic
        self.basic_conv1_w = torch.from_numpy(np.fromfile(dir + "\\BasicConv1\\w.bin",dtype=np.float32)).view(32,3,3,3)
        self.basic_conv1_b = torch.from_numpy(np.fromfile(dir + "\\BasicConv1\\b.bin",dtype=np.float32))
        self.basic_conv2_w = torch.from_numpy(np.fromfile(dir + "\\BasicConv2\\w.bin", dtype=np.float32)).view(64,32,3, 3)
        self.basic_conv2_b = torch.from_numpy(np.fromfile(dir + "\\BasicConv2\\b.bin", dtype=np.float32))
        self.basic_conv3_w = torch.from_numpy(np.fromfile(dir + "\\BasicConv3\\w.bin", dtype=np.float32)).view(512,512,3, 3)
        self.basic_conv3_b = torch.from_numpy(np.fromfile(dir + "\\BasicConv3\\b.bin", dtype=np.float32))
        #resblock1,64,64
        self.resblock1_w1 = torch.from_numpy(np.fromfile(dir + "\\ResBlock1\\w1.bin",dtype=np.float32)).view(64,64,3,3)
        self.resblock1_b1 = torch.from_numpy(np.fromfile(dir + "\\ResBlock1\\b1.bin",dtype=np.float32))
        self.resblock1_w2 = torch.from_numpy(np.fromfile(dir + "\\ResBlock1\\w2.bin", dtype=np.float32)).view(32,32,3,3)
        self.resblock1_b2 = torch.from_numpy(np.fromfile(dir + "\\ResBlock1\\b2.bin", dtype=np.float32))
        self.resblock1_w3 = torch.from_numpy(np.fromfile(dir + "\\ResBlock1\\w3.bin", dtype=np.float32)).view(32,32,3,3)
        self.resblock1_b3 = torch.from_numpy(np.fromfile(dir + "\\ResBlock1\\b3.bin", dtype=np.float32))
        self.resblock1_w4 = torch.from_numpy(np.fromfile(dir + "\\ResBlock1\\w4.bin", dtype=np.float32)).view(64,64,1,1)
        self.resblock1_b4 = torch.from_numpy(np.fromfile(dir + "\\ResBlock1\\b4.bin", dtype=np.float32))
        #resblock2,128,128
        self.resblock2_w1 = torch.from_numpy(np.fromfile(dir + "\\ResBlock2\\w1.bin", dtype=np.float32)).view(128,128,3,3)
        self.resblock2_b1 = torch.from_numpy(np.fromfile(dir + "\\ResBlock2\\b1.bin", dtype=np.float32))
        self.resblock2_w2 = torch.from_numpy(np.fromfile(dir + "\\ResBlock2\\w2.bin", dtype=np.float32)).view(64,64,3,3)
        self.resblock2_b2 = torch.from_numpy(np.fromfile(dir + "\\ResBlock2\\b2.bin", dtype=np.float32))
        self.resblock2_w3 = torch.from_numpy(np.fromfile(dir + "\\ResBlock2\\w3.bin", dtype=np.float32)).view(64,64,3,3)
        self.resblock2_b3 = torch.from_numpy(np.fromfile(dir + "\\ResBlock2\\b3.bin", dtype=np.float32))
        self.resblock2_w4 = torch.from_numpy(np.fromfile(dir + "\\ResBlock2\\w4.bin", dtype=np.float32)).view(128,128,1,1)
        self.resblock2_b4 = torch.from_numpy(np.fromfile(dir + "\\ResBlock2\\b4.bin", dtype=np.float32))
        #resblock3,256,256
        self.resblock3_w1 = torch.from_numpy(np.fromfile(dir + "\\ResBlock3\\w1.bin", dtype=np.float32)).view(256,256,3,3)
        self.resblock3_b1 = torch.from_numpy(np.fromfile(dir + "\\ResBlock3\\b1.bin", dtype=np.float32))
        self.resblock3_w2 = torch.from_numpy(np.fromfile(dir + "\\ResBlock3\\w2.bin", dtype=np.float32)).view(128,128,3,3)
        self.resblock3_b2 = torch.from_numpy(np.fromfile(dir + "\\ResBlock3\\b2.bin", dtype=np.float32))
        self.resblock3_w3 = torch.from_numpy(np.fromfile(dir + "\\ResBlock3\\w3.bin", dtype=np.float32)).view(128,128,3,3)
        self.resblock3_b3 = torch.from_numpy(np.fromfile(dir + "\\ResBlock3\\b3.bin", dtype=np.float32))
        self.resblock3_w4 = torch.from_numpy(np.fromfile(dir + "\\ResBlock3\\w4.bin", dtype=np.float32)).view(256,256,1,1)
        self.resblock3_b4 = torch.from_numpy(np.fromfile(dir + "\\ResBlock3\\b4.bin", dtype=np.float32))


    def forward(self,x):
        x=FusedBasicConv(x,self.basic_conv1_w,self.basic_conv1_b,2)
        x=FusedBasicConv(x,self.basic_conv2_w,self.basic_conv2_b,2)
        x,_=FusedResBlock(x,64,64,
                          self.resblock1_w1,self.resblock1_w2,self.resblock1_w3,self.resblock1_w4,
                          self.resblock1_b1,self.resblock1_b2,self.resblock1_b3,self.resblock1_b4)
        x,_=FusedResBlock(x,128,128,
                          self.resblock2_w1, self.resblock2_w2, self.resblock2_w3, self.resblock2_w4,
                          self.resblock2_b1, self.resblock2_b2, self.resblock2_b3, self.resblock2_b4)
        x,feat1=FusedResBlock(x,256,256,
                          self.resblock3_w1, self.resblock3_w2, self.resblock3_w3, self.resblock3_w4,
                          self.resblock3_b1, self.resblock3_b2, self.resblock3_b3, self.resblock3_b4)
        feat2=FusedBasicConv(x,self.basic_conv3_w,self.basic_conv3_b,1)
        return feat1,feat2


#############################################################################
class MyYolo:
    def __init__(self):
        self.backbone=MyCSPdarknet53_tiny()
        #conv_forP5
        self.conv_forP5_w=None
        self.conv_forP5_b=None
        #headP4
        self.yolo_headP4_w1=None
        self.yolo_headP4_w2=None
        self.yolo_headP4_b1=None
        self.yolo_headP4_b2=None
        #headP5
        self.yolo_headP5_w1 = None
        self.yolo_headP5_w2 = None
        self.yolo_headP5_b1 = None
        self.yolo_headP5_b2 = None
        #upsample
        self.upsample_w=None
        self.upsample_b=None

    def forward(self,x):
        feat1,feat2=self.backbone.forward(x)
        P5=FusedBasicConv(feat2,self.conv_forP5_w,self.conv_forP5_b,s=1,p=0)         #K=1
        #out0
        out0=FusedBasicConv(P5,self.yolo_headP5_w1,self.yolo_headP5_b1,s=1)
        out0=F.conv2d(input=out0,weight=self.yolo_headP5_w2,bias=self.yolo_headP5_b2,stride=1,padding=0)
        #P5_Upsample
        P5_Upsample=FusedBasicConv(P5,self.upsample_w,self.upsample_b,s=1,p=0)       #K=1
        P5_Upsample=F.upsample(input=P5_Upsample,scale_factor=2,mode='nearest')
        #P4
        P4=torch.cat([P5_Upsample,feat1],dim=1)
        #out1
        out1=FusedBasicConv(P4,self.yolo_headP4_w1,self.yolo_headP4_b1,s=1)
        out1=F.conv2d(input=out1,weight=self.yolo_headP4_w2,bias=self.yolo_headP4_b2,stride=1,padding=0)
        #
        return out0,out1

    def load_weight(self,dir):
        self.backbone.load_weight(dir)
        #conv_forP5
        self.conv_forP5_w=torch.from_numpy(np.fromfile(dir+"\\conv_forP5\\w.bin",dtype=np.float32)).view(256,512,1,1)
        self.conv_forP5_b=torch.from_numpy(np.fromfile(dir+"\\conv_forP5\\b.bin",dtype=np.float32))
        #upsample
        self.upsample_w=torch.from_numpy(np.fromfile(dir+"\\upsample\\w.bin",dtype=np.float32)).view(128,256,1,1)
        self.upsample_b=torch.from_numpy(np.fromfile(dir+"\\upsample\\b.bin",dtype=np.float32))
        #head4
        self.yolo_headP4_w1 = torch.from_numpy(np.fromfile(dir + "\\yolo_headP4\\w1.bin", dtype=np.float32)).view(256,384,3,3)
        self.yolo_headP4_b1 = torch.from_numpy(np.fromfile(dir + "\\yolo_headP4\\b1.bin", dtype=np.float32))
        self.yolo_headP4_w2 = torch.from_numpy(np.fromfile(dir + "\\yolo_headP4\\w2.bin", dtype=np.float32)).view(75,256,1,1)
        self.yolo_headP4_b2 = torch.from_numpy(np.fromfile(dir + "\\yolo_headP4\\b2.bin", dtype=np.float32))
        #head5
        self.yolo_headP5_w1 = torch.from_numpy(np.fromfile(dir + "\\yolo_headP5\\w1.bin", dtype=np.float32)).view(512,256,3,3)
        self.yolo_headP5_b1 = torch.from_numpy(np.fromfile(dir + "\\yolo_headP5\\b1.bin", dtype=np.float32))
        self.yolo_headP5_w2 = torch.from_numpy(np.fromfile(dir + "\\yolo_headP5\\w2.bin", dtype=np.float32)).view(75,512,1,1)
        self.yolo_headP5_b2 = torch.from_numpy(np.fromfile(dir + "\\yolo_headP5\\b2.bin", dtype=np.float32))

#############################################################################
from yolo import *

def backbone_test():
    x=torch.randn(1,3,416,416)
    yolo=YoloBody(anchors_mask=[[3,4,5],[1,2,3]],num_classes=20,phi=0,pretrained=False)
    yolo.load_state_dict(torch.load("yolov4_tiny_weights_voc.pth"))
    yolo.eval()
    mybackbone=MyCSPdarknet53_tiny()
    mybackbone.load_weight("folded_weights")
    #
    f11,f12=mybackbone.forward(x)
    f21,f22=yolo.backbone.forward(x)
    print(torch.max(torch.abs(f11-f21)))
    print(torch.max(torch.abs(f12-f22)))
    x.numpy().tofile("x.bin")
    f11.numpy().tofile("feat1.bin")
    f12.numpy().tofile("feat2.bin")

def yolo_test():
    #输入
    x=torch.randn(10,3,416,416)
    #baseline
    yolo=YoloBody(anchors_mask=[[3,4,5], [1,2,3]],num_classes=20,phi=0,pretrained=False)
    yolo.load_state_dict(torch.load("yolov4_tiny_weights_voc.pth"))
    yolo.eval()
    #ours
    myyolo=MyYolo()
    myyolo.load_weight("folded_weights")
    #test
    o1,o2=myyolo.forward(x)
    r1,r2=yolo.forward(x)
    #compare
    print(torch.max(torch.abs(r1-o1)))
    print(torch.max(torch.abs(r2-o2)))

import os
def BasicConvTest(dir,h,w,k,s,p,c,n,i):
    weight=torch.from_numpy(np.fromfile(dir+"\\w.bin",dtype=np.float32)).view(n,c,k,k)
    bias=torch.from_numpy(np.fromfile(dir+"\\b.bin",dtype=np.float32))
    x=torch.randn(1,c,h,w)
    out=FusedBasicConv(x=x,w=weight,b=bias,s=s,p=p)
    if not os.path.exists("BasicConv{}".format(i)):
        os.mkdir("BasicConv{}".format(i))
    weight.numpy().tofile("BasicConv{}\\w.bin".format(i))
    bias.numpy().tofile("BasicConv{}\\b.bin".format(i))
    x.numpy().tofile("BasicConv{}\\x.bin".format(i))
    out.numpy().tofile("BasicConv{}\\out.bin".format(i))

def ResBlockTest(dir,h,w,c,n,i):
    w1=torch.from_numpy(np.fromfile(dir+"\\w1.bin",dtype=np.float32)).view(n,c,3,3)
    b1=torch.from_numpy(np.fromfile(dir+"\\b1.bin",dtype=np.float32))
    w2=torch.from_numpy(np.fromfile(dir+"\\w2.bin",dtype=np.float32)).view(n//2,n//2,3,3)
    b2=torch.from_numpy(np.fromfile(dir+"\\b2.bin",dtype=np.float32))
    w3=torch.from_numpy(np.fromfile(dir+"\\w3.bin",dtype=np.float32)).view(n//2,n//2,3,3)
    b3=torch.from_numpy(np.fromfile(dir+"\\b3.bin",dtype=np.float32))
    w4=torch.from_numpy(np.fromfile(dir+"\\w4.bin",dtype=np.float32)).view(n,n,1,1)
    b4=torch.from_numpy(np.fromfile(dir+"\\b4.bin",dtype=np.float32))
    x=torch.randn(1,c,h,w)
    #创建保存的文件夹
    dir="ResBlock{}".format(i)
    if not os.path.exists(dir):
        os.mkdir(dir)
    y,feat=FusedResBlock(x,n=n,c=c,w1=w1,w2=w2,w3=w3,w4=w4,b1=b1,b2=b2,b3=b3,b4=b4)
    #保存
    x.numpy().tofile(dir+"\\x.bin")
    y.numpy().tofile(dir+"\\y.bin")
    feat.numpy().tofile(dir+"\\feat.bin")
    w1.numpy().tofile(dir+"\\w1.bin")
    b1.numpy().tofile(dir+"\\b1.bin")
    w2.numpy().tofile(dir+"\\w2.bin")
    b2.numpy().tofile(dir+"\\b2.bin")
    w3.numpy().tofile(dir+"\\w3.bin")
    b3.numpy().tofile(dir+"\\b3.bin")
    w4.numpy().tofile(dir+"\\w4.bin")
    b4.numpy().tofile(dir+"\\b4.bin")

if __name__=="__main__":
    BasicConvTest("folded_weights\\BasicConv1",416,416,3,2,1,3,32,1)
    BasicConvTest("folded_weights\\BasicConv2",208,208,3,2,1,32,64,2)
    BasicConvTest("folded_weights\\BasicConv3",13,13,3,1,1,512,512,3)
    ResBlockTest("folded_weights\\ResBlock1",104,104,64,64,1)
    ResBlockTest("folded_weights\\ResBlock2",52,52,128,128,2)
    ResBlockTest("folded_weights\\ResBlock3",26,26,256,256,3)
    backbone_test()
    yolo_test()
