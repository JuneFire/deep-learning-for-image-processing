# https://blog.csdn.net/Zhang_Chen_/article/details/108433594
"""
感受野大小的计算采用，Top To Down的方式， 即先计算最深层在前一层上的感受野，然后逐渐传递到第一层，使用的公式可以表示如下：
RF=1
# 待计算的feature map上的感受野大小

for layer in (top layer to down layer):
	RF = ((RF - 1) * stride) + kernel size
	# stride 为每一层kernel stride的累乘
	# kernel size为当前层的kernel size
"""

net_struct = {
    'vgg16': {'net': [[3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1],
                      [2, 2, 0], [3, 1, 1], [3, 1, 1], [5, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1],
                      [2, 2, 0]],
              'name': ['conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'pool2', 'conv3_1', 'conv3_2', 'conv3_3',
                       'pool3', 'conv4_1', 'conv4_2', 'conv4_4', 'pool4', 'conv5_1', 'conv5_2', 'conv5_3', 'pool5']}
}

imsize = 224


def outFromIn(isz, net, layernum):
    totstride = 1
    insize = isz
    for layer in range(layernum):
        fsize, stride, pad = net[layer]
        outsize = (insize - fsize + 2 * pad) / stride + 1
        insize = outsize
        totstride = totstride * stride
    return outsize, totstride


def inFromOut(net, layernum):
    RF = 1
    for layer in reversed(range(layernum)):
        fsize, stride, pad = net[layer]
        RF = ((RF - 1) * stride) + fsize
    return RF


if __name__ == '__main__':
    print("layer output sizes given image = %dx%d" % (imsize, imsize))
    for net in net_struct.keys():
        print('************net structrue name is %s**************' % net)
        for i in range(len(net_struct[net]['net'])):
            p = outFromIn(imsize, net_struct[net]['net'], i + 1)
            rf = inFromOut(net_struct[net]['net'], i + 1)
            print("Layer Name = %s, Output size = %3d, Stride = % 3d, RF size = %3d" % (
                net_struct[net]['name'][i], p[0], p[1], rf))
