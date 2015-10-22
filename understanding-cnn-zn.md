---
layout: page
permalink: /understanding-cnn/
---

<a name='vis'></a>

(this page is currently in draft form)

## Visualizing what ConvNets learn

Several approaches for understanding and visualizing Convolutional Networks have been developed in the literature, partly as a response the common criticism that the learned features are not interpretable. In this section we briefly survey some of these approaches and related work.

有许多方法可以用来理解并可视化CNN网络， 这些方法的目的是为了更好的解释CNN学习到的特征表达。 
这个章节主要目的是简要介绍当前存在的一些可视化方法。

### Visualizing the activations and first-layer weights

**Layer Activations**. The most straight-forward visualization technique is to show the activations of the network during the forward pass. For ReLU networks, the activations usually start out looking relatively blobby and dense, but as the training progresses the activations usually become more sparse and localized. One dangerous pitfall that can be easily noticed with this visualization is that some activation maps may be all zero for many different inputs, which can indicate *dead* filters, and can be a symptom of high learning rates.

<div class="fig figcenter fighighlight">
  <img src="/assets/cnnvis/act1.jpeg" width="49%">
  <img src="/assets/cnnvis/act2.jpeg" width="49%">
  <div class="figcaption">
    Typical-looking activations on the first CONV layer (left), and the 5th CONV layer (right) of a trained AlexNet looking at a picture of a cat. Every box shows an activation map corresponding to some filter. Notice that the activations are sparse (most values are zero, in this visualization shown in black) and mostly local.
  </div>
</div>

1. 最直接的可视化Layer Activations的方法是在forward pass中显示网络激励network activation.
2. 对于ReLu网络， activations开始的时候看起来相对blobby, dense，但经过训练过程后， activations变得更加sparse， localized.
3. 在可视化中能发现的一个危险的信号是， 对于不同的inputs某些activations会变成zeros, 这暗示着filters不起作用了，也表示训练时设置的learning rate过大。
4. 在上面这个图例中显示了对AlexNet网络输入一个cat图片时 1st CONV layer和5th CONV layer的activation。 图中每个box代表一个activation map对应于一些filter。 注意到这些activations都是sparse的(大部分值是zero，在图中显示为黑色)，而且大部分是局部的。


**Conv/FC Filters.** The second common strategy is to visualize the weights. These are usually most interpretable on the first CONV layer which is looking directly at the raw pixel data, but it is possible to also show the filter weights deeper in the network. The weights are useful to visualize because well-trained networks usually display nice and smooth filters without any noisy patterns. Noisy patterns can be an indicator of a network that hasn't been trained for long enough, or possibly a very low regularization strength that may have led to overfitting.

<div class="fig figcenter fighighlight">
  <img src="/assets/cnnvis/filt1.jpeg" width="49%">
  <img src="/assets/cnnvis/filt2.jpeg" width="49%">
  <div class="figcaption">
    Typical-looking filters on the first CONV layer (left), and the 2nd CONV layer (right) of a trained AlexNet. Notice that the first-layer weights are very nice and smooth, indicating nicely converged network. The color/grayscale features are clustered because the AlexNet contains two separate streams of processing, and an apparent consequence of this architecture is that one stream develops high-frequency grayscale features and the other low-frequency color features. The 2nd CONV layer weights are not as interpretible, but it is apparent that they are still smooth, well-formed, and absent of noisy patterns.
  </div>
</div>

1. 另一种可视化的策略是可视化weights. 最容易解释的是在第一层CONV layer处可以直接看到raw pixel data, 但也可以在更深的layer中显示filter weights.
2. 可视化weights是十分有用的， 因为训练得比较好的网络通畅会有比较smooth filters without any noisy pattern.
3. Noisy pattern 可以作为一个信号，表示一个网络并没有训练足够长时间，或者regularization力度不够，导致模型overfitting.
4. 上面这个图里中展示了 1st CONV layer和2nd CONV layer的filters. 可以看到1st的weights十分平滑，表面网络converged. 在AlexNet中， 这些filter weights代表了两种：高频grayscale和低频color features。 2nd CONV layer的weights则不那么容易解释，但是看起来它们仍然sooth, well-formed而且noisy patterns.

### Retrieving images that maximally activate a neuron

Another visualization technique is to take a large dataset of images, feed them through the network and keep track of which images maximally activate some neuron. We can then visualize the images to get an understanding of what the neuron is looking for in its receptive field. One such visualization (among others) is shown in [Rich feature hierarchies for accurate object detection and semantic segmentation](http://arxiv.org/abs/1311.2524) by Ross Girshick et al.:

<div class="fig figcenter fighighlight">
  <img src="/assets/cnnvis/pool5max.jpeg" width="100%">
  <div class="figcaption">
    Maximally activating images for some POOL5 (5th pool layer) neurons of an AlexNet. The activation values and the receptive field of the particular neuron are shown in white. (In particular, note that the POOL5 neurons are a function of a relatively large portion of the input image!) It can be seen that some neurons are responsive to upper bodies, text, or specular highlights.
  </div>
</div>

One problem with this approach is that ReLU neurons do not necessarily have any semantic meaning by themselves. Rather, it is more appropriate to think of multiple ReLU neurons as the basis vectors of some space that represents in image patches. In other words, the visualization is showing the patches at the edge of the cloud of representations, along the (arbitrary) axes that correspond to the filter weights. This can also be seen by the fact that neurons in a ConvNet operate linearly over the input space, so any arbitrary rotation of that space is a no-op. This point was further argued in [Intriguing properties of neural networks](http://arxiv.org/abs/1312.6199) by Szegedy et al., where they perform a similar visualization along aribtrary directions in the representation space.

1. 另一种可视化的方法是 选择一个大的图片数据集， 然后将他们送到network中跟踪，看哪些图片能最大化激励某些神经元。 通过这种方式可以分析网络中各个神经元的可接受域receptive field.
2. 上图中的例子显示了 某些POOL5 layer中最大化激励的典型图片。 可以看到，一些神经元对uper bodies, text, specular highlights敏感。

### Embedding the codes with t-SNE 

ConvNets can be interpreted as gradually transforming the images into a representation in which the classes are separable by a linear classifier. We can get a rough idea about the topology of this space by embedding images into two dimensions so that their low-dimensional representation has approximately equal distances than their high-dimensional representation. There are many embedding methods that have been developed with the intuition of embedding high-dimensional vectors in a low-dimensional space while preserving the pairwise distances of the points. Among these, [t-SNE](http://lvdmaaten.github.io/tsne/) is one of the best-known methods that consistently produces visually-pleasing results.

To produce an embedding, we can take a set of images and use the ConvNet to extract the CNN codes (e.g. in AlexNet the 4096-dimensional vector right before the classifier, and crucially, including the ReLU non-linearity). We can then plug these into t-SNE and get 2-dimensional vector for each image. The corresponding images can them be visualized in a grid:

<div class="fig figcenter fighighlight">
  <img src="/assets/cnnvis/tsne.jpeg" width="100%">
  <div class="figcaption">
    t-SNE embedding of a set of images based on their CNN codes. Images that are nearby each other are also close in the CNN representation space, which implies that the CNN "sees" them as being very similar. Notice that the similarities are more often class-based and semantic rather than pixel and color-based. For more details on how this visualization was produced the associated code, and more related visualizations at different scales refer to <a href="http://cs.stanford.edu/people/karpathy/cnnembed/">t-SNE visualization of CNN codes</a>.
  </div>
</div>

1. CNN可以看做是 逐渐地将图片转换为一种适当的表达representation, 使得线性分类器可以区分这些图片。 一种想法是， 可以通过embedding将图片放入two dimensions， 从而他们的low-dimensional representation比high-dimensional representation有更近似相同的距离。
2. 有许多的embedding方法 用于将高纬度representation映射到低纬度的空间，同时保持样本间的距离。 比较有名的方法包括t-SNE.
3. 具体的，可以通过训练好的CNN网络提取图片的feature (如AlexNet中可以得到4096-dim的feature vector), 然后可以使用t-SNE方法得到每个图片的2-dim的vector， 于是对应的图片可以在一个grid中可视化。
4. 上面图例中展示了t-SNE的一个例子。 grid中相邻的图片在原始的4096-dim空间中也相近。 这里的相近是指class-based, semantic层面的，而不是pixel, color-based.

### Occluding parts of the image

### Visualizing the data gradient

### DeconvNets and related

### Reconstructing original images based on CNN Codes

### Plotting performance as a function of image attributes

## Fooling ConvNets

## Comparing ConvNets to Human labelers
