# NeRF: Neural Radiance Field in 3D Vision, A Comprehensive Review
**论文:** [https://arxiv.org/pdf/2210.00379.pdf](https://arxiv.org/pdf/2210.00379.pdf)
![在这里插入图片描述](https://img-blog.csdnimg.cn/9e5f2353bc5342cb9b7a22ae9a6ccd0d.png)
&emsp;&emsp;NeRF体素渲染和训练过程( a ) 说明了待合成图像中各个像素的采样点选择。( b ) 说明了使用NeRF MLP在采样点处产生密度和颜色。( c ) 和 ( d ) 示出了通过体素渲染使用沿着相关联的相机射线的颜色和密度的单个像素颜色的生成，以及分别与地面真实像素颜色的比较。
# 摘要

> `Abstract`
> &emsp;&emsp;Neural Radiance Field (NeRF), a new novel view synthesis with implicit scene representation has taken the ﬁeld of Computer Vision by storm. As a novel view synthesis and 3D reconstruction method, NeRF models ﬁnd applications in robotics, urban mapping, autonomous navigation, virtual reality/augmented reality, and more. Since the original paper by Mildenhall et al., more than 250 preprints were published, with more than 100 eventually being accepted in tier one Computer Vision Conferences. Given NeRF popularity and the current interest in this research area, we believe it necessary to compile a comprehensive survey of NeRF papers from the past two years, which we organized into both architecture, and application based taxonomies. We also provide an introduction to the theory of NeRF based novel view synthesis, and a benchmark comparison of the performance and speed of key NeRF models. By creating this survey, we hope to introduce new researchers to NeRF, provide a helpful reference for inﬂuential works in this ﬁeld, as well as motivate future research directions with our discussion section.
> `Index Terms`
> &emsp;&emsp;Neural Radiance Field, NeRF, Computer Vision Survey, Novel View Synthesis, Neural Rendering, 3D Reconstruction, Multi-view
译文：
&emsp;&emsp;神经辐射场 (NeRF)，一种具有隐式场景表示的新型视图合成，已经席卷了计算机视觉领域。作为一种新颖的视图合成和 3D 重建方法，NeRF 模型在机器人技术、城市测绘、自主导航、虚拟现实/增强现实等领域都有应用。自 Mildenhall 等人的原始论文以来，已发表了 250 多份预印本，其中 100 多份最终被一级计算机​​视觉会议接受。鉴于 NeRF 的受欢迎程度和当前对该研究领域的兴趣，我们认为有必要对过去两年的 NeRF 论文进行全面调查，我们将其组织成基于架构和基于应用程序的分类法。我们还介绍了基于 NeRF 的新颖视图合成理论，以及关键 NeRF 模型的性能和速度的基准比较。通过创建这项调查，我们希望向 NeRF 介绍新的研究人员，为该领域的有影响力的工作提供有用的参考，并通过我们的讨论部分激发未来的研究方向。
神经辐射场、NeRF、计算机视觉调查、新视图合成、神经渲染、3D 重建、多视图

# 文章脉络
• 第1节引言，简要介绍NeRF的知名度，工作原理和工作领域。（这里不做介绍）
• 第2节介绍了现有的NeRF，解释了NeRF`体素渲染背后的理论`，介绍了常用的`数据集`和`质量评估指标`。
• 第3节是本文的核心，并`介绍了有影响力的NeRF`出版物，并包含了我们为组织这些作品而创建的分类法。它的小节详细介绍了过去两年中提出的NeRF创新的不同家族，以及NeRF模型在各种计算机视觉任务中的最新应用。
• 第4节和第5节(`总结与展望`)讨论了潜在的未来研究方向和应用，并总结了调查。
# 神经辐射场 (NeRF) 概念
## 理论介绍
&emsp;&emsp;NeRF模型以其基本形式将三维场景表示为由神经网络近似的辐射场。辐射场描述了场景中每个点和每个观看方向的颜色和体积密度。这写为:
$$F\left( x,\theta ,\varphi \right) →\left( c,\sigma \right) ,(1)$$
&emsp;&emsp;其中$x = (x，y，z)$ 是场景内坐标，$(θ，φ)$ 表示方位角和极视角，$c = (r，g，b)$ 表示颜色，$σ$表示体积密度。该5D函数由一个或多个多层预加速器 (MLP) 近似，有时表示为f Θ。两个视角 $(θ，φ)$通常由$d = (dx，dy，dz)$表示，这是一个3D笛卡尔单位向量。通过将 $σ$ (体积密度 (即场景的内容) 的预测限制为与观看方向无关)，该神经网络表示被约束为多视图一致，而允许颜色$c$取决于观看方向和场景内坐标。在基线NeRF模型中，这是通过将MLP设计为两个阶段来实现的。
&emsp;&emsp;第一阶段作为输入$x$并输出 $σ$ 和高维特征向量 (在原始论文中256)。在第二阶段，特征向量然后与观看方向$d$连接，并传递给额外的MLP，该MLP输出$c$。我们注意到Mildenhall等人 [1] 认为$σ$ MLP和$c$ MLP是同一神经网络的两个分支，但是许多后来的作者认为它们是两个独立的MLP网络，这是我们从这一点开始遵循的惯例。从广义上讲，使用经过训练的NeRF模型进行的新颖视图合成如下。
- 对于正在合成的图像中的每个像素，通过场景发送相机光线并生成一组采样点 (参见图1中的 (a))。
-  对于每个采样点，使用观看方向和采样位置来提取局部颜色和密度，由NeRF MLP(s) 计算 (参见图1中的 (b))。
- 使用体绘制从这些颜色和密度产生图像 (参见图1中的 (c))。

&emsp;&emsp;更详细地说，给定体积密度和颜色函数，使用体积渲染来获得任何相机射线$r(t) = o+td$的颜色$C(r)$，相机位置$o$和观看方向$d$使用
$$
C(r)=\int_{t_1}^{t_2}{T(t)·\sigma(r(t))·c(r(t),d)·dt},(2)
$$
&emsp;&emsp;其中$T(t)$ 是累积透射率，表示光线从$t_1$传播到$t$而不被拦截的概率，由
$$
T(t)=e^{-\int_{t}^{t_1}{\sigma (r(u))·du}},(3)
$$
&emsp;&emsp;$C(r)$通过待合成图像的每个像素。这个积分可以用数值计算。最初的实现 [1] 和大多数后续方法使用了非确定性分层抽样方法，将射线分成$N$个等间距的仓，并从每个仓中均匀抽取一个样本。然后，等式 (2) 可以近似为
$$
\hat{C}\left( r \right) =\sum_{i=1}^N{\alpha _iT_ic_i}\,,\,where\quad T_i=e^{-\sum_{j=1}^{i-1}{\sigma _j\delta _j}},(4)
$$
&emsp;&emsp;$\delta _i$是从样本$i$到样本$i+1$的距离。$(\sigma_i,c_i)$是根据NeRF MLP(s) 计算的在给定射线的采样点$i$上评估的密度和颜色。$α_i$在采样点$i$处合成$alpha$的透明度/不透明度由
$$
\alpha_i = 1-e^{\sigma_i\delta_i},(5)
$$
&emsp;&emsp;可以使用累积的透射率计算射线的预期深度为
$$
d(r)=\int_{t_1}^{t_2}{T(t)·\sigma(r(t))·t·dt},(6)
$$
&emsp;&emsp;这可以近似于方程 (4) 近似方程 (2) 和 (3)
$$
\hat{D}(r) = \sum_{i=1}^{N}{\alpha_it_iT_i},(7)
$$
&emsp;&emsp;某些深度正则化方法使用预期的深度来将密度限制为场景表面的类似delta的函数，或增强深度平滑度。
&emsp;&emsp;对于每个像素，使用平方误差光度损失来优化MLP参数。在整个图像上，这是由
$$
L = \sum_{r\in R}{|| \hat{C}(r)-C_{gt}(r)||_2^2},(8)
$$
&emsp;&emsp;其中，$C_{gt}(r)$ 是与$r$相关联的训练图像的像素的地面真实颜色,$R$是与待合成图像相关联的射线批次。

## 数据集
数据集地址：[https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)
- **Synthetic NeRFDataset**
- **Local Light Field Fusion (LLFF) Dataset**
- **DTU Dataset**
- **ScanNet Dataset**
- **Tanks andTemples Dataset**
- **ShapeNet Dataset**

##  评估指标
&emsp;&emsp;在标准设置中，通过NeRF进行新颖的视图综合使用基准的视觉质量评估指标。这些指标试图评估具有 (完全参考) 或不具有 (无参考) 地面真实图像的单个图像的质量。峰值信噪比 (PSNR)，结构相似性指数度量 (SSIM) [31]，学习的感知图像补丁相似性 (LPIPS) [32] 是迄今为止NeRF文献中最常用的。
### PSNR
&emsp;&emsp;PSNR是一种无参考质量评估指标：
$$
PSNR(I)=10·log_{10}{\frac{MAX(I)^2}{MSE(I)}},(10)
$$
&emsp;&emsp;其中$MAX(I)$ 是图像中的最大可能像素值 (对于8位整数255)，并且$MSE(I)$ 是在所有颜色通道上计算的像素方向均方误差。$PNSR$也通常用于信号处理的其他领域，并且被很好地理解。
## SSIM
&emsp;&emsp;SSIM是一个完整的参考质量评估指标,对于单个小块
$$
SSIM(x,y) = \frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\mu_x^2+\mu_y^2+C_2)},(11)
$$
&emsp;&emsp;其中$C_i = (K_iL)^2$，L是像素的动态范围 (对于8bit整数255)，并且$K_1 = 0.01$，$K_2 = 0.03$是由原始作者选择的常数。我们注意到，在原始论文 [31] 中，有 (12) 给出的$SSIM$的更一般形式。在11 × 11圆形对称高斯加权窗口内计算局部统计量$\mu^,s$, $\sigma^,s$，权重$w_i$的标准差为1.5，并归一化为1。这些是由给出的，没有损失概括
$$
\mu_x = \sum_i{w_ix_i},(12)
$$
$$
\sigma_x=(\sum_iw_i(x_i-\mu_x)^2)^{\frac{1}{2}},(13)
$$
$$
\sigma_{xy}=\sum_iw_i(x_i-\mu_x)(y_i-\mu_y),(14)
$$
&emsp;&emsp;其中$x_i$，$y_i$分别是从参考图像和评估图像中采样的像素。在实践中，对整个图像的$SSIM$分数进行平均。
## LPIPS
&emsp;&emsp;$LPIPS$是使用学习的卷积特征的完整参考质量评估指标。得分由多层特征图的加权像素$MSE$给出。
$$
LPIPS(x,y)=\sum_l^L\frac{1}{H_lW_l}\sum_{h,w}^{H_l,W_l}{||w_l\odot x_{hw}^{l}-y_{hw}^{l}||}_2^2,(15)
$$
&emsp;&emsp;$x_{hw}^{l}$和$y_{hw}^{l}$是参考和评估图像在像素宽度w，像素高度h和层l处的特征。Hl和Wl是相应层处的特征图高度和宽度。最初的$LPIPS$论文使用SqueezeNet [444]，VGG [34] 和AlexNet [35] 作为特征提取主干。原始纸张使用了五层。原始作者提供了微调和从头开始的配置，但实际上，已按原样使用预先训练的网络。
# 基于方法分类的NeRF变体
![在这里插入图片描述](https://img-blog.csdnimg.cn/813f26f935174975a60bade2bcd0a7c8.png)

> NeRF创新论文的分类学。这些论文是结合引用和GitHub星级来选择的。我们注意到，基于MLP的无速度模型严格来说并不是NeRF模型。由于它们最近的流行以及它们与某些基于速度的NeRF模型的相似性，我们决定将它们包括在此分类学树中。

![在这里插入图片描述](https://img-blog.csdnimg.cn/63c6e684e0bc4c329f89f3190093a8d1.png)
> 在synthetic NeRF dataset上各个NeRF模型的性能比较

## 基本原理
## 改进训练和推理渲染速度
## 少量拍摄/稀疏训练视图NeRF
## (潜在) 条件NeRF
## 解绑场景与场景构图
## 姿态估计
## 相邻方法
##  基于NeRF应用的分类树
### Urban 
![在这里插入图片描述](https://img-blog.csdnimg.cn/c5ad6204955548c58c39f3bbc5a5c3c3.png)

 - Urban Radiance Field：[https://urban-radiance-fields.github.io/](https://urban-radiance-fields.github.io/)
 - BlockNeRF：[https://waymo.com/intl/zh-cn/research/block-nerf/](https://waymo.com/intl/zh-cn/research/block-nerf/)
 - MegaNeRF：[https://meganerf.cmusatyalab.org/](https://meganerf.cmusatyalab.org/) *代码已开源*
 - BungeeNeRF：[https://city-super.github.io/citynerf/](https://city-super.github.io/citynerf/)*代码已开源*
 - S-NeRF：[https://github.com/esa/snerf](https://github.com/esa/snerf)*代码已开源*

&emsp;&emsp;城市辐射场旨在应用基于NeRF的视图合成和基于激光雷达数据的稀疏多视图图像对城市环境进行3D重建。除了标准的光度损失外，他们还使用基于激光雷达的深度损失$L_{depth}$和视线损失$L_{sight}$，以及基于skybox的分割损失Lseg。这些是由
$$L_{depth} = E[(z − \hat{z}^2)], (30)$$
$$
L_{sight}=E\left[ \int_{_{t_1}}^{t_2}{\left( w\left( t \right) −\delta \left( z \right) \right)^2dt} \right] ,(31)
$$
$$
L_{seg}=E\left[ S_i\left(r\int_{_{t_1}}^{t_2}{\left( w\left( t \right) −\delta \left( z \right) \right)^2dt} \right)\right] ,(32)
$$

&emsp;&emsp;$w(t)$ 定义为公式 (3) 中定义的$T(t)σ(t)$。$z$和 $ζ$ 分别是激光雷达测量深度和估计深度 (6)。$Δ (z)$ 是狄拉克 $δ$函数。如果光线穿过第i图像中的天空像素，则$S_i(r) = 1$，其中天空像素通过预训练模型 [145] 被分割，否则为0。深度损失迫使估计的深度 “$z$” 与激光雷达获取的深度匹配。视线损失迫使辐射集中在测量深度的表面。分割损失迫使沿着光线穿过天空像素的点样本具有零密度。通过在体积渲染过程中从NeRF模型中提取点云来执行3D重建。为虚拟相机中的每个像素投射射线。然后，使用估计的深度将点云放置在3D场景中。泊松表面重建用于从此生成的点云构建3D网格。
#### Mega-NeRF
&emsp;&emsp;Mega-NeRF [129] (2021年12月) 根据航空无人机图像进行了大规模的城市重建。Mega-NeRF使用NeRF [67] 逆球参数化将前景与背景分开。但是，作者通过使用更适合鸟瞰图的椭球来扩展该方法。他们还将nerf-w [65] 的每个图像外观嵌入代码纳入了他们的模型中。他们将大型城市场景划分为单元，每个单元由其自己的NeRF模块表示，并仅在具有潜在相关像素的图像上训练每个模块。对于渲染，该方法还将密度和颜色的粗略渲染缓存到八叉树中。在进行动态飞越的渲染过程中，会快速生成粗略的初始视图，并通过其他几轮模型采样进行动态优化。该模型的性能优于基准基准，例如NeRF [67]，基于COLMAP的MVS重建 [2]，并制作了令人印象深刻的飞行视频。
#### Block-NeRFs
&emsp;&emsp;Block-NeRFs [128] (2022年2月) 从280万街道级图像中进行了基于城市尺度NeRF的重建。如此大规模的户外数据集带来了诸如瞬态外观和对象之类的问题。每个单独的BlockNeRF都是通过使用其IPE和nerf-w [65] 通过使用其外观潜在代码优化在mip-NeRF [36] 上构建的。此外，作者在NeRF训练期间使用语义分割来掩盖诸如行人和汽车之类的瞬态对象。使用传递性函数 (3) 和NeRF MLP生成的密度值对可见性MLP进行并行训练。这些用于丢弃低可见度Block-nerf。邻里被分为几个街区，并在其上训练了一个街区神经。为这些块分配了重叠，并从重叠的Block-nerf中采样图像，并在外观代码匹配优化后使用反向距离加权进行合成。

#### Others
&emsp;&emsp;在一流会议上发布的其他方法，例如S-NeRF [131] (2021年4月)，BungeeNeRF [130] (2021年12月)，也涉及城市3D重建和视图合成，尽管来自遥感图像。

### Human Body
![在这里插入图片描述](https://img-blog.csdnimg.cn/8bcab6b0e2b0471fa735e16502a96471.png)
 - Nerﬁes:https://nerfies.github.io/[添加链接描述](https://nerfies.github.io/)*代码已开源*
 - HyperNeRF:https://hypernerf.github.io/[添加链接描述](https://hypernerf.github.io/)*代码已开源*
 - HeadNeRF:[https://hy1995.top/HeadNeRF-Project/](https://hy1995.top/HeadNeRF-Project/)*代码已开源*
 - Neural Body:[https://zju3dv.github.io/neuralbody/](https://zju3dv.github.io/neuralbody/)*代码已开源*
 - HumanNeRF:[https://grail.cs.washington.edu/projects/humannerf/](https://grail.cs.washington.edu/projects/humannerf/)*代码已开源*
 - DoubleField:[http://www.liuyebin.com/dbfield/dbfield.html](http://www.liuyebin.com/dbfield/dbfield.html)
 - Animatable NeRF:[https://zju3dv.github.io/animatable_nerf/](https://zju3dv.github.io/animatable_nerf/)*代码已开源*

&emsp;&emsp;人体神经体应用NeRF体绘制来渲染视频中移动姿势的人类。作者首先使用输入视频来锚定基于顶点的可变形人体模型 ($SMPL$[146])。在每个顶点上，作者附加了一个16维潜码$Z$。然后使用人体姿势参数$S$ (最初是在训练过程中从视频中估计的，可以在推理过程中输入) 来变形人体模型。该模型在网格中进行体素化，然后由3D CNN处理，该3D CNN在每个占用的体素处提取128维潜码 (特征)。对于任何3D点$x$，首先将其转换为$SMPL$坐标系，然后通过插值提取128维潜码/特征 $ψ$。这被传递给alpha MLP。RGB MLP还采用了3D坐标 $γ_x(x)$的位置编码和查看方向$γ_d(d)$和外观潜码$l_t$(考虑视频中的每帧差异)。
$$
\sigma \left( x \right) =M_\sigma \left( \varphi \left( x|Z，S \right) \right) ,(33)
$$
$$
c \left( x \right) =M_\sigma \left( \varphi \left( x|Z，S \right)  ,γ_x(x),γ_d(d),lt\right),(34)
$$

&emsp;&emsp;标准的NeRF方法与运动物体作斗争，而神经体的网格变形方法能够在框架之间和姿势之间进行插值。来自2021/2022顶级会议的最先进模型，如动画NeRF [144] (2021年5月) 、DoubleField [143] (2021年6月) 、HumanNeRF [141] (2022年1月) 、郑等人 [142] (2022年3月) 也在这一主题上进行了创新。

### Image Processing
![在这里插入图片描述](https://img-blog.csdnimg.cn/0b30115d62074ff79263cb1be8c79228.png)
 - ClipNeRF:[https://arxiv.org/abs/2112.05139](https://arxiv.org/abs/2112.05139)
 - EditNeRF:[http://editnerf.csail.mit.edu/](http://editnerf.csail.mit.edu/)*代码已开源*
 - CodeNeRF:[https://arxiv.org/abs/2109.01750](https://arxiv.org/abs/2109.01750)*代码已开源*
 - CoNeRF:[https://conerf.github.io/](https://conerf.github.io/)*代码已开源*
 - Semantic-NeRF:[https://shuaifengzhi.com/Semantic-NeRF/](https://shuaifengzhi.com/Semantic-NeRF/)*代码已开源*
 - NeSF:[https://nesf3d.github.io/](https://nesf3d.github.io/)*代码已开源*
 - Fig-NeRF:[https://fig-nerf.github.io/](https://fig-nerf.github.io/)
 - HDR/Tone Mapping -- RawNeRF:[https://bmild.github.io/rawnerf/](https://bmild.github.io/rawnerf/)*代码已开源*
 - HDR/Tone Mapping -- HDR-NeRF:[https://shsf0817.github.io/hdr-nerf/](https://shsf0817.github.io/hdr-nerf/)*代码已开源*
 - Denoising/Deblurring -- RawNeRF:[https://bmild.github.io/rawnerf/](https://bmild.github.io/rawnerf/)*代码已开源*
 - Denoising/Deblurring -- DeblurNeRF:[https://limacv.github.io/deblurnerf/](https://limacv.github.io/deblurnerf/)*代码已开源*
 - Denoising/Deblurring -- NaN:[https://noise-aware-nerf.github.io/](https://noise-aware-nerf.github.io/) *code coming soon*
 
 
 #### RawNeRF
&emsp;&emsp;Mildenhall等人创建了RawNeRF [38] (2021年11月)，使Mip-NeRF [36] 适应高动态范围 (HDR) 图像视图合成和去噪。RawNeRF使用原始线性图像作为训练数据在线性颜色空间中渲染。这允许改变曝光和色调映射曲线，本质上是在NeRF渲染后应用后处理，而不是直接使用后处理图像作为训练数据。使用相对MSE损耗对噪声2噪声的HDR路径跟踪进行训练 [147]，由
$$
L=\sum_{i=1}^N{\left( \frac{\hat{y}_i-y_i}{sg\left( \hat{y}_i \right) +\epsilon} \right)^2},(35)
$$
&emsp;&emsp;其中sg(.) 表示梯度停止 (将其参数视为具有零梯度的常数)。RawNeRF由可变曝光图像监督，NeRF模型的 “曝光” 由训练图像的快门速度以及每通道学习的缩放因子缩放。它在夜间和弱光场景渲染和去噪方面取得了令人印象深刻的效果。RawNeRF特别适合光线不足的场景。关于基于NeRF的去噪，NaN [135] (4月10日) 也探索了这一新兴的研究领域。
#### HDR-NeRF
&emsp;&emsp;与RawNeRF同时，Xin等人的HDR-NeRF [133] (2021年11月) 也致力于HDR视图合成。但是，与RawNeRF中的原始线性图像相反，HDR-NeRF通过使用具有可变曝光时间的低动态范围训练图像来进行HDR视图合成。RawNeRF建模了HDR辐射e(r) ∈ [0，∞) 取代了 (1) 中的标准c(r)。使用三个MLP摄像机响应函数 (每个颜色通道一个) f，该辐射被映射到彩色c。这些代表了典型的摄像机制造商依赖的线性和非线性后处理。HDR-NeRF的性能大大优于基线NeRF和nerf-w [65] 在低动态范围 (LDR) 重建中，并在HDR重建方面获得了较高的视觉评估得分。

#### Semantic-NeRF
&emsp;&emsp;Semantic-NeRF [70] (2021年3月) 是一个能够合成新颖视图的语义标签的NeRF模型。这是使用附加的与方向无关的MLP (分支) 完成的，该MLP将位置和密度MLP特征作为输入，并生成逐点语义标签$s$。语义标签也是通过体绘制生成的。
$$
S(r) = \sum _i^N{T_i\alpha_is_i},(36)
$$
&emsp;&emsp;语义标签通过分类交叉熵损失进行监督。该方法能够利用稀疏语义标签 (10% 标记) 训练数据进行训练，以及从像素噪声和区域/实例噪声中恢复语义标签。该方法还获得了良好的标签超分辨率结果，并且标签传播 (来自稀疏点逐点注释)，并且可以用于多视图语义融合，优于非深度学习方法。NeSF [132] (2021年11月)。Fig-NeRF [68] 也解决了这个问题。

### Surface Reconstruction
![在这里插入图片描述](https://img-blog.csdnimg.cn/61fc9a0f3ea144fd8ad943de29077a27.png)
 - Neural Surface:[http://geometry.cs.ucl.ac.uk/projects/2021/neuralmaps/](http://geometry.cs.ucl.ac.uk/projects/2021/neuralmaps/)*代码已开源*
 - Neural RGB-D:[https://dazinovic.github.io/neural-rgbd-surface-reconstruction/](https://dazinovic.github.io/neural-rgbd-surface-reconstruction/)*代码已开源*
 - UNISURF:[https://moechsle.github.io/unisurf/](https://moechsle.github.io/unisurf/)*代码已开源*
 
&emsp;&emsp; NeRF模型的场景几何是隐式的，隐藏在神经网络中。但是，对于某些应用，需要更明确的表示，例如3D网格。对于基线NeRF，可以通过评估和阈值密度MLP来提取粗略的几何形状。本小节中介绍的方法使用了创新的场景表示策略，从而改变了密度MLP的基本行为。

#### UNISERF
&emsp;&emsp;UNISERF [138] (2021年4月) 通过用离散占用函数$\sigma(x) = 1$替换由 (5) 给出的离散化体绘制方程中使用的第i个样本点处的alpha值ai来重建场景曲面，并且在自由空间中$\sigma(x) = 0$。此占用函数也由MLP计算，基本上被替换为体积密度。然后通过沿射线寻根来取回表面。UNISURF优于基准方法，包括在基线NeRF模型中使用密度阈值，以及IDR [148]。占用MLP可用于定义场景的显式表面几何形状。特斯拉 (Tesla) [149] 最近的一个研讨会显示，自动驾驶模块的3D理解是由一个类似NeRF的占用网络驱动的。
Signed distance functions(SDF) 给出3D点到它们定义的表面的有符号距离 (即，如果在对象内部，则为负距离，如果在外部，则为正距离)。它们通常用于计算机图形学中，以定义曲面，这些曲面是函数的零集。SDF可通过根查找用于表面重建，并可用于定义整个场景几何形状。
#### Neural Surface
&emsp;&emsp;神经表面 (NeuS) [136] (2021年6月) 模型像基线NeRF模型一样执行体积渲染。但是，它使用带符号距离函数来定义场景几何形状。它将MLP的密度输出部分替换为输出有符号距离函数值的MLP。然后将在体绘制方程 (2) 中替换 $σ(t)$的密度 $ρ(t)$ 构造为
$$
\rho \left( t \right) =max\left( \frac{\frac{−d\varPhi}{dt}\left( f\left( r\left( t \right) \right) \right)}{ \varPhi \left( f\left( r\left( t \right) \right) \right)} ,0 \right) ,(37)
$$
&emsp;&emsp;其中 $\varPhi(·)$是$s$形函数，其导数$\frac{d\varPhi}{dt}$是logistic密度分布。作者已经证明了他们的模型优于基线NeRF，并为他们的方法和基于SDF的场景密度的实现提供了理论和实验依据。此方法与UNISERF并发，并且在DTU数据集上优于它 [26]。与UNISURF一样，可以使用在SDF上执行根查找来定义场景的显式表面几何形状。
#### Others
&emsp;&emsp;Azinovic等人的同时工作[137] (2021年4月) 也用截断的SDF MLP代替了密度MLP。相反，他们将像素颜色计算为采样颜色的加权总和
$$
C\left( r \right) =\frac{\sum_{i=1}^N{w_ic_i}}{\sum_{i=1}^N{w_i}},(38)
$$
其中$w_i$由$s$形函数的乘积给出，$s$形函数由
$$
w_i = \varPhi(\frac{D_i}{tr}) · \varPhi(-\frac{D_i}{tr})	,(39)
$$
&emsp;&emsp;其中tr是截断距离，它截断了距离单个曲面太远的任何SDF值。为了考虑可能的多个射线表面相交，随后的截断区域加权为零，并且对像素颜色没有贡献。作者还使用nerf-w [65] 中的每帧外观潜在代码来解释白平衡和曝光变化。作者通过在截断的SDF MLP上使用行进立方体 [150] 重建了场景的三角形网格，并在ScanNet[28] 和私有合成数据集上获得了干净的重建结果 (但由于未提供DTU结果，因此无法直接与UNISERF和NeuS进行比较)。

# 总结与展望
## 速度
&emsp;&emsp;NeRF模型的训练速度和推理速度都很慢。目前，基于速度的NeRF/nerf邻接模型使用三个主要范例。它们要么
1) be baked(通过评估已经训练好的NeRF模型，并缓存/baking其结果)，要么
2) 通过使用额外的学习的体素/空间树特征将学习的场景特征与颜色和密度mlp分开，或者
3) 在不使用mlp的情况下直接对学习到的体素特征执行体积渲染。

&emsp;&emsp;通过在体绘制过程中使用更高级的方法 (例如空位跳跃和早期射线终) 可以提高速度。然而，方法1) 并没有通过其设计来改善训练时间。方法2) 和3) 需要额外的内存，因为与小NeRF mlp相比，基于体素/空间树的场景特征具有更大的内存占用。当前，InstantNGP [48] 通过使用多分辨率散列位置编码作为额外的学习特征，显示了最大的希望，该模型可以用微小而高效的mlp准确地表示场景。这允许极快的训练和推理。Instant-NGP模型还可以在图像压缩和神经SDF场景表示中找到应用。或者，对于方法3)，学习体素网格的因式张量 [55] 表示将学习体素特征的记忆需求降低了两个数量级，并且也是一个可行的研究领域。我们期望未来基于速度的方法遵循方法2) 和3)，对于这些方法，具有高度影响力的研究方向将是改善这些额外学习的场景特征的数据结构和设计。

## 精度
&emsp;&emsp;为了提高质量，我们发现nerf-w [65] 的每个图像瞬态潜代码和外观代码的实现具有影响力。在并发GRAF上也发现了类似的想法 [61]。这些潜在代码使NeRF模型可以控制每个图像的照明/颜色变化以及场景内容的微小变化。在NeRF基础知识上，我们发现mip-NeRF [36] 最有影响力，因为锥跟踪IPE不同于任何以前的NeRF实现。基于mipNeRF的Ref-NeRF，然后进一步改进了与视图相关的外观。Ref-nerf是基于质量的NeRF研究的优秀现代基线。具体到图像处理，RawNeRF [38] 和DeblurNeRF [134] 的创新可以结合Ref-NeRF作为基础来构建极高质量的去噪/去模糊NeRF模型。

## 关于姿势估计和稀疏视图
&emsp;&emsp;鉴于NeRF研究的现状，我们认为非大满贯姿势估计是一个已解决的问题。使用COLMAP[2] 包的SfM被大多数NeRF数据集用来提供近似姿势，这对于大多数NeRF研究来说已经足够了。BA还可以用来在训练中联合优化NeRF模型和姿势。基于NeRF的SLAM是一个相对未开发的研究领域。iMAP [71] 和Nice-SLAM [72] 提供了出色的基于NeRF的SLAM框架，可以集成更快，质量更好的NeRF模型。
&emsp;&emsp;稀疏视图/少镜头NeRF使用预先训练的CNN从多视图图像中提取2D/3D特征。有些还使用SfM的点云进行额外的监督。我们相信这些模型中的许多已经实现了很少拍摄的目标 (2-10个视图)。通过使用更高级的特征提取主干，可以实现进一步的小改进。我们相信研究的一个关键领域将是结合稀疏视图方法和快速方法来实现可部署到移动设备的实时NeRF模型。

## 应用
&emsp;&emsp;我们相信NeRF的直接应用是新颖的视图合成和城市环境以及人类化身的3D重建。通过促进从密度mlp中提取3D网格，点云或SDF并集成更快的NeRF模型，可以进行进一步的改进。城市环境特别要求将环境划分为单独的小场景，每个场景都由一个特定于小场景的NeRF模型表示。针对城市规模模型的基于速度的NeRF模型，对单独的场景特征进行烘焙或学习是一个有趣的研究方向。对于人类化身，可以分离特定于视图的照明 (例如RefNeRF[37]) 的模型的集成将对虚拟现实和3D图形等应用非常有利。NeRF还在诸如去噪，去模糊，上采样，压缩和图像编辑等基本图像处理任务中找到应用，随着越来越多的计算机视觉从业者采用NeRF模型，我们预计在不久的将来这些领域将有更多的创新。

## 结论
&emsp;&emsp;自Mildenhall等人的原始论文以来，NeRF模型在速度，质量和训练视图要求方面取得了巨大进步，改善了原始模型的所有弱点。NeRF模型已在城市制图/建模/摄影测量，图像编辑/标记，图像处理以及人类化身和城市环境的3D重建和视图合成中找到了应用。本调查详细讨论了技术改进和应用，在完成调查期间，我们注意到对NeRF模型的兴趣日益浓厚，预印本和出版物的数量也在不断增加。NeRF是新颖视图合成，3D重建和神经渲染的令人兴奋的新范例。通过提供此调查，我们希望向该领域介绍更多的计算机视觉从业人员，为现有的NeRF模型提供有用的参考，并通过我们的讨论来激励未来的研究。我们很高兴看到神经辐射领域的未来技术创新和应用。

