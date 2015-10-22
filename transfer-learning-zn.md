---
layout: page
permalink: /transfer-learning/
---

(These notes are currently in draft form and under development)

Table of Contents:

- [Transfer Learning](#tf)
- [Additional References](#add)

<a name='tf'></a>
## Transfer Learning

In practice, very few people train an entire Convolutional Network from scratch (with random initialization), because it is relatively rare to have a dataset of sufficient size. Instead, it is common to pretrain a ConvNet on a very large dataset (e.g. ImageNet, which contains 1.2 million images with 1000 categories), and then use the ConvNet either as an initialization or a fixed feature extractor for the task of interest. The three major Transfer Learning scenarios look as follows:

- **ConvNet as fixed feature extractor**. Take a ConvNet pretrained on ImageNet, remove the last fully-connected layer (this layer's outputs are the 1000 class scores for a different task like ImageNet), then treat the rest of the ConvNet as a fixed feature extractor for the new dataset. In an AlexNet, this would compute a 4096-D vector for every image that contains the activations of the hidden layer immediately before the classifier. We call these features **CNN codes**. It is important for performance that these codes are ReLUd (i.e. thresholded at zero) if they were also thresholded during the training of the ConvNet on ImageNet (as is usually the case). Once you extract the 4096-D codes for all images, train a linear classifier (e.g. Linear SVM or Softmax classifier) for the new dataset.
- **Fine-tuning the ConvNet**. The second strategy is to not only replace and retrain the classifier on top of the ConvNet on the new dataset, but to also fine-tune the weights of the pretrained network by continuing the backpropagation. It is possible to fine-tune all the layers of the ConvNet, or it's possible to keep some of the earlier layers fixed (due to overfitting concerns) and only fine-tune some higher-level portion of the network. This is motivated by the observation that the earlier features of a ConvNet contain more generic features (e.g. edge detectors or color blob detectors) that should be useful to many tasks, but later layers of the ConvNet becomes progressively more specific to the details of the classes contained in the original dataset. In case of ImageNet for example, which contains many dog breeds, a significant portion of the representational power of the ConvNet may be devoted to features that are specific to differentiating between dog breeds.

>
1. 实际中， 很少人直接去重头开始训练整个CNN网络(比如利用随机初始化random initialization), 因为很难获得一个有sufficient size的数据集。 通常情况是 先在一个大的数据集比如ImageNet上pretrain一个CNN网络， 然后利用这个pretrained的网络来作为初始值或fixed feature extrator，用于其他的tasks。 

一般有三种迁移学习的场景：
>
1. CNN作为fixed feature extractor。 利用ImageNet上pretrained的CNN网络， 去掉最后一层FC7 layer， 然后将剩余的网络层级作为一个fixed feature extractor, 用于新的dataset。 比如在AlexNet中，每个图片可以计算一个4096-dim的feature vector, 这个vector包含了之前隐藏层的activation输出。 这个feature vector通常称为CNN codes。 对于实际性能来说， 最重要的是这些codes是ReLUd，通过了阈值处理的。 对于新的dataset，先提取4096-dim的codes之后，就可以利用linear SVM或者softmax分类器来训练线性分类器了。
2. 对CNN进行Fine-tuning。  第二种策略是 不但对CNN的顶层进行replace和retrain, 还对之前pretrained的CNN网络之前层的weights进行fine-tune(利用backpropagation)。 对CNN的所有层进行fine-tuning是可能的， 或者固定某些ealier的中间hidden layers，然后fine-tune其他的hidden layers。  这种策略的基础是， CNN网络的earlier features包含了更多的generic features(如edge detectors, color blob detectors)， 这些generic features 对不同的tasks更加有效， 而之后的later layers则变得更加的specific, 对original dataset(如ImageNet)更加的依赖(不适应新的dataset)。 比如在ImageNet中包含了许多dog breeds, 因此pretrained的CNN网络则对区分不同的dog breeds有更加specific的特性。

**Pretrained models**. Since modern ConvNets take 2-3 weeks to train across multiple GPUs on ImageNet, it is common to see people release their final ConvNet checkpoints for the benefit of others who can use the networks for fine-tuning. For example, the Caffe library has a [Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo) where people share their network weights.

> 由于现在的CNN网络在ImageNet数据集上一般需要2-3周来训练(利用multiple GPUs)， 所以我们经常见到学者们公开他们自己训练的CNN网络的checkpoints，这些checkpoints可以被其他人用于fine-tuning. 

**When and how to fine-tune?** How do you decide what type of transfer learning you should perform on a new dataset? This is a function of several factors, but the two most important ones are the size of the new dataset (small or big), and its similarity to the original dataset (e.g. ImageNet-like in terms of the content of images and the classes, or very different, such as microscope images). Keeping in mind that ConvNet features are more generic in early layers and more original-dataset-specific in later layers, here are some common rules of thumb for navigating the 4 major scenarios:

1. *New dataset is small and similar to original dataset*. Since the data is small, it is not a good idea to fine-tune the ConvNet due to overfitting concerns. Since the data is similar to the original data, we expect higher-level features in the ConvNet to be relevant to this dataset as well. Hence, the best idea might be to train a linear classifier on the CNN codes.
2. *New dataset is large and similar to the original dataset*. Since we have more data, we can have more confidence that we won't overfit if we were to try to fine-tune through the full network.
3. *New dataset is small but very different from the original dataset*. Since the data is small, it is likely best to only train a linear classifier. Since the dataset is very different, it might not be best to train the classifier form the top of the network, which contains more dataset-specific features. Instead, it might work better to train the SVM classifier from activations somewhere earlier in the network.
4. *New dataset is large and very different from the original dataset*. Since the dataset is very large, we may expect that we can afford to train a ConvNet from scratch. However, in practice it is very often still beneficial to initialize with weights from a pretrained model. In this case, we would have enough data and confidence to fine-tune through the entire network.

> 何时以及怎样进行fine-tune, 有两个需要考虑的因素：1) new dataset的大小 (small or big); 2) new dataset与original dataset的相似程度。 始终需要注意的是： CNN features通常在earlier layers时更加generic, 而在later layers上则更加original-dataset-specific. 以下的四个场景可以归纳出一些common rules:
>
1. "new dataset较小， 而且与original dataset相似"。 由于数据集小，所以fine-tune 整个CNN网络不可取，会导致overfitting. 由于data与original dataset相似， 因此CNN网络中的higher-level features会与new dataset更相关。 此种情况的最好方式是 先对new dataset中的images提取CNN codes，然后训练linear classifiers.
2. "new dataset大， 而且与original dataset相似"。 因为有更多的与original dataset相似的数据，因此可以考虑fine-tune整个CNN网络， 不会造成overfitting.
3. "new dataset较小， 而且与original dataset不同"。 由于数据集较小，所以最好训练linear classifier. 由于new dataset与original dataset不同， 因此直接在CNN的top layer上训练不可取。 更好的方案是， 提取ealier layers的activations，然后训练linear SVM classifier。
4. "new dataset大，而且与original dataset不同"。 这种情况下我们可以直接在new dataset上训练CNN网络，同时我们可以利用original dataset的weights作为new dataset的CNN网络的初始weights。 

**Practical advice**. There are a few additional things to keep in mind when performing Transfer Learning:

- *Constraints from pretrained models*. Note that if you wish to use a pretrained network, you may be slightly constrained in terms of the architecture you can use for your new dataset. For example, you can't arbitrarily take out Conv layers from the pretrained network. However, some changes are straight-forward: Due to parameter sharing, you can easily run a pretrained network on images of different spatial size. This is clearly evident in the case of Conv/Pool layers because their forward function is independent of the input volume spatial size (as long as the strides "fit"). In case of FC layers, this still holds true because FC layers can be converted to a Convolutional Layer: For example, in an AlexNet, the final pooling volume before the first FC layer is of size [6x6x512]. Therefore, the FC layer looking at this volume is equivalent to having a Convultional Layer that has receptive field size 6x6, and is applied with padding of 0.
- *Learning rates*. It's common to use a smaller learning rate for ConvNet weights that are being fine-tuned, in comparison to the (randomly-initialized) weights for the new linear classifier that computes the class scores of your new dataset. This is because we expect that the ConvNet weights are relatively good, so we don't wish to distort them too quickly and too much (especially while the new Linear Classifier above them is being trained from random initialization).

> 实际建议
>
1. "pretrained models的限制"。 如果要利用pretrained model, 需要考虑到pretrained network的结构。 比如，不能随意从pretrained CNN中修改layers. 但是，有些修改是直观的： 由于有parameter sharing, pretrained network可以用于不同大小的iamges上。 原因是在CONV/POOL layers中， 他们的forward function是独立于input volume spatial size (strides "fit"). 而对于FC layer， 以上仍然成立， 由于FC layer可以转化成CNN layer。 比如在AlexNet中， 在第一个FC layer之前的final pooling volumn大小是[6x6x512]. 因此， FC layer可以将这个volumn等效看做CNN layer， 有receptive field size 6x6, with padding 0.
2. "learning rate"。 在fine-tune时， 一般的lr要比pretrained CNN时设置的更小，而且要比最后一层random initilized的classifier layer的weights更小。 因为我们期望CNN的weights相对教好，因此不希望改变他们太快太多， 尤其是在他们之后的new linear classifier layer采用random initilization时。

<a name='tf'></a>
## Additional References

- [CNN Features off-the-shelf: an Astounding Baseline for Recognition](http://arxiv.org/abs/1403.6382) trains SVMs on features from ImageNet-pretrained ConvNet and reports several state of the art results.
- [DeCAF](http://arxiv.org/abs/1310.1531) reported similar findings in 2013. The framework in this paper (DeCAF) was a Python-based precursor to the C++ Caffe library.
- [How transferable are features in deep neural networks?](http://arxiv.org/abs/1411.1792) studies the transfer learning performance in detail, including some unintuitive findings about layer co-adaptations.