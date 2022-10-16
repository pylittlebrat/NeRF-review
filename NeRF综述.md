# NeRF: Neural Radiance Field in 3D Vision, A Comprehensive Review
**论文:** [https://arxiv.org/pdf/2210.00379.pdf](https://arxiv.org/pdf/2210.00379.pdf)
![在这里插入图片描述](https://img-blog.csdnimg.cn/9e5f2353bc5342cb9b7a22ae9a6ccd0d.png)
&emsp;&emsp;NeRF体素渲染和训练过程( a ) 说明了待合成图像中各个像素的采样点选择。( b ) 说明了使用NeRF MLP在采样点处产生密度和颜色。( c ) 和 ( d ) 示出了通过体素渲染使用沿着相关联的相机射线的颜色和密度的单个像素颜色的生成，以及分别与地面真实像素颜色的比较。
# 摘要

> `Abstract`
> &emsp;&emsp;Neural Radiance Field (NeRF), a new novel view synthesis with implicit scene representation has taken the ﬁeld of Computer Vision by storm. As a novel view synthesis and 3D reconstruction method, NeRF models ﬁnd applications in robotics, urban mapping, autonomous navigation, virtual reality/augmented reality, and more. Since the original paper by Mildenhall et al., more than 250 preprints were published, with more than 100 eventually being accepted in tier one Computer Vision Conferences. Given NeRF popularity and the current interest in this research area, we believe it necessary to compile a comprehensive survey of NeRF papers from the past two years, which we organized into both architecture, and application based taxonomies. We also provide an introduction to the theory of NeRF based novel view synthesis, and a benchmark comparison of the performance and speed of key NeRF models. By creating this survey, we hope to introduce new researchers to NeRF, provide a helpful reference for inﬂuential works in this ﬁeld, as well as motivate future research directions with our discussion section.
> `Index Terms`
> &emsp;&emsp;Neural Radiance Field, NeRF, Computer Vision Survey, Novel View Synthesis, Neural Rendering, 3D Reconstruction, Multi-view<br>
译文：
&emsp;&emsp;神经辐射场 (NeRF)，一种具有隐式场景表示的新型视图合成，已经席卷了计算机视觉领域。作为一种新颖的视图合成和 3D 重建方法，NeRF 模型在机器人技术、城市测绘、自主导航、虚拟现实/增强现实等领域都有应用。自 Mildenhall 等人的原始论文以来，已发表了 250 多份预印本，其中 100 多份最终被一级计算机​​视觉会议接受。鉴于 NeRF 的受欢迎程度和当前对该研究领域的兴趣，我们认为有必要对过去两年的 NeRF 论文进行全面调查，我们将其组织成基于架构和基于应用程序的分类法。我们还介绍了基于 NeRF 的新颖视图合成理论，以及关键 NeRF 模型的性能和速度的基准比较。通过创建这项调查，我们希望向 NeRF 介绍新的研究人员，为该领域的有影响力的工作提供有用的参考，并通过我们的讨论部分激发未来的研究方向。<br>
神经辐射场、NeRF、计算机视觉调查、新视图合成、神经渲染、3D 重建、多视图

# 文章脉络
• 第1节引言，简要介绍NeRF的知名度，工作原理和工作领域。（这里不做介绍）<br>
• 第2节介绍了现有的NeRF，解释了NeRF`体素渲染背后的理论`，介绍了常用的`数据集`和`质量评估指标`。<br>
• 第3节是本文的核心，并`介绍了有影响力的NeRF`出版物，并包含了我们为组织这些作品而创建的分类法。它的小节详细介绍了过去两年中提出的NeRF创新的不同家族，以及NeRF模型在各种计算机视觉任务中的最新应用。<br>
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
$$C(r)=\int_{t_1}^{t_2}{T(t)·\sigma(r(t))·c(r(t),d)·dt},(2)$$
&emsp;&emsp;其中$T(t)$ 是累积透射率，表示光线从$t_1$传播到$t$而不被拦截的概率，由
$$T(t)=e^{-\int_{t}^{t_1}{\sigma (r(u))·du}},(3)$$
&emsp;&emsp;$C(r)$通过待合成图像的每个像素。这个积分可以用数值计算。最初的实现 [1] 和大多数后续方法使用了非确定性分层抽样方法，将射线分成$N$个等间距的仓，并从每个仓中均匀抽取一个样本。然后，等式 (2) 可以近似为

$$\hat{C}\left( r \right) =\sum_{i=1}^N{\alpha _iT_ic_i}\,,\,where\quad T_i=e^{-\sum_{j=1}^{i-1}{\sigma _j\delta _j}},(4)$$

&emsp;&emsp;$\delta _i$是从样本$i$到样本$i+1$的距离。$(\sigma_i,c_i)$是根据NeRF MLP(s) 计算的在给定射线的采样点$i$上评估的密度和颜色。$α_i$在采样点$i$处合成$alpha$的透明度/不透明度由

$$\alpha_i = 1-e^{\sigma_i\delta_i},(5)$$

&emsp;&emsp;可以使用累积的透射率计算射线的预期深度为

$$d(r)=\int_{t_1}^{t_2}{T(t)·\sigma(r(t))·t·dt},(6)$$

&emsp;&emsp;这可以近似于方程 (4) 近似方程 (2) 和 (3)

$$\hat{D}(r) = \sum_{i=1}^{N}{\alpha_it_iT_i},(7)$$

&emsp;&emsp;某些深度正则化方法使用预期的深度来将密度限制为场景表面的类似delta的函数，或增强深度平滑度。
&emsp;&emsp;对于每个像素，使用平方误差光度损失来优化MLP参数。在整个图像上，这是由

$$L = \sum_{r\in R}{|| \hat{C}(r)-C_{gt}(r)||_2^2},(8)$$

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
$$PSNR(I)=10·log_{10}{\frac{MAX(I)^2}{MSE(I)}},(10)$$
&emsp;&emsp;其中$MAX(I)$ 是图像中的最大可能像素值 (对于8位整数255)，并且$MSE(I)$ 是在所有颜色通道上计算的像素方向均方误差。$PNSR$也通常用于信号处理的其他领域，并且被很好地理解。
## SSIM
&emsp;&emsp;SSIM是一个完整的参考质量评估指标,对于单个小块
$$SSIM(x,y) = \frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\mu_x^2+\mu_y^2+C_2)},(11)$$
&emsp;&emsp;其中$C_i = (K_iL)^2$，L是像素的动态范围 (对于8bit整数255)，并且$K_1 = 0.01$，$K_2 = 0.03$是由原始作者选择的常数。我们注意到，在原始论文 [31] 中，有 (12) 给出的$SSIM$的更一般形式。在11 × 11圆形对称高斯加权窗口内计算局部统计量$\mu^,s$, $\sigma^,s$，权重$w_i$的标准差为1.5，并归一化为1。这些是由给出的，没有损失概括
$$\mu_x = \sum_i{w_ix_i},(12)$$
$$\sigma_x=(\sum_iw_i(x_i-\mu_x)^2)^{\frac{1}{2}},(13)$$
$$\sigma_{xy}=\sum_iw_i(x_i-\mu_x)(y_i-\mu_y),(14)$$
&emsp;&emsp;其中$x_i$，$y_i$分别是从参考图像和评估图像中采样的像素。在实践中，对整个图像的$SSIM$分数进行平均。
## LPIPS
&emsp;&emsp;$LPIPS$是使用学习的卷积特征的完整参考质量评估指标。得分由多层特征图的加权像素$MSE$给出。

$$LPIPS(x,y)=\sum_l^L\frac{1}{H_lW_l}\sum_{h,w}^{H_l,W_l}{||w_l\odot x_{hw}^{l}-y_{hw}^{l}||}_2^2,(15)$$

&emsp;&emsp;$x_{hw}^{l}$和$y_{hw}^{l}$是参考和评估图像在像素宽度w，像素高度h和层l处的特征。Hl和Wl是相应层处的特征图高度和宽度。最初的$LPIPS$论文使用SqueezeNet [444]，VGG [34] 和AlexNet [35] 作为特征提取主干。原始纸张使用了五层。原始作者提供了微调和从头开始的配置，但实际上，已按原样使用预先训练的网络。
# 基于方法分类的NeRF变体
![在这里插入图片描述](https://img-blog.csdnimg.cn/813f26f935174975a60bade2bcd0a7c8.png)

> NeRF创新论文的分类学。这些论文是结合引用和GitHub星级来选择的。我们注意到，基于MLP的无速度模型严格来说并不是NeRF模型。由于它们最近的流行以及它们与某些基于速度的NeRF模型的相似性，我们决定将它们包括在此分类学树中。

![在这里插入图片描述](https://img-blog.csdnimg.cn/63c6e684e0bc4c329f89f3190093a8d1.png)
> 在synthetic NeRF dataset上各个NeRF模型的性能比较

## 基本原理
![在这里插入图片描述](https://img-blog.csdnimg.cn/003b7ae84aaa4b6d9703d580e11b355d.png)
- mip-NeRF(2021):[https://jonbarron.info/mipnerf/](https://jonbarron.info/mipnerf/)*代码已开源*
- mip-NeRF 360 (2022):[https://jonbarron.info/mipnerf360/](https://jonbarron.info/mipnerf360/)*代码已开源*
- Ref-NeRF(2021):[https://dorverbin.github.io/refnerf/](https://dorverbin.github.io/refnerf/)*代码已开源*
- RawNeRF(2021):[https://bmild.github.io/rawnerf/](https://bmild.github.io/rawnerf/)*代码已开源*
- Nerfies (2020):[https://nerfies.github.io/](https://nerfies.github.io/)*代码已开源*
- HyperNeRF(2021):[https://hypernerf.github.io/](https://hypernerf.github.io/)*代码已开源*
- CodeNeRF (2021):[https://sites.google.com/view/wbjang/home/codenerf](https://sites.google.com/view/wbjang/home/codenerf)*代码已开源*
- DS-NeRF (2021):[https://www.cs.cmu.edu/~dsnerf/](https://www.cs.cmu.edu/~dsnerf/)*代码已开源*
- NerfingMVS (2021):[https://weiyithu.github.io/NerfingMVS/](https://weiyithu.github.io/NerfingMVS/)*代码已开源*
- Urban Radiance Field (2021):[https://urban-radiance-fields.github.io/](https://urban-radiance-fields.github.io/)
- PointNeRF (2022):[https://xharlie.github.io/projects/project_sites/pointnerf/](https://xharlie.github.io/projects/project_sites/pointnerf/)*代码已开源*

## 改进训练和推理渲染速度
![在这里插入图片描述](https://img-blog.csdnimg.cn/4478c168885f46f6af7150523d79c202.png)
- NSVF (2020)：[https://lingjie0206.github.io/papers/NSVF/](https://lingjie0206.github.io/papers/NSVF/)*代码已开源*
- AutoInt (2020)：[https://github.com/shichence/AutoInt](https://github.com/shichence/AutoInt)*代码已开源*
- Instant-NGP (2022)：[https://nvlabs.github.io/instant-ngp/](https://nvlabs.github.io/instant-ngp/)*代码已开源*
- SNeRG (2020)：[https://phog.github.io/snerg/](https://phog.github.io/snerg/)*代码已开源*
- Plenoctree (2021)：[https://alexyu.net/plenoctrees/](https://alexyu.net/plenoctrees/)*代码已开源*
- FastNeRF (2021)：[https://microsoft.github.io/FastNeRF/](https://microsoft.github.io/FastNeRF/)
(code:[https://github.com/houchenst/FastNeRF](https://github.com/houchenst/FastNeRF)?)
- KiloNeRF (2021)：[https://github.com/creiser/kilonerf](https://github.com/creiser/kilonerf)*代码已开源*
- Plenoxels (2021)：[https://alexyu.net/plenoxels/](https://alexyu.net/plenoxels/)*代码已开源*
- DVGO (2021)：[https://sunset1995.github.io/dvgo/](https://sunset1995.github.io/dvgo/)*代码已开源*
- TensorRF (2021)：[https://github.com/NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT)*代码已开源*

&emsp;&emsp;Mildenhall等人 [1] 基于原NeRF实现中，为了提高计算效率，使用了分层渲染。在数值积分 (2) 期间，天真的渲染将需要密集地评估沿每个相机射线的所有查询点处的mlp。在他们提出的方法中，他们使用两个网络来表示场景，一个粗略，一个精细。粗网络的输出用于为精细网络选择采样点，从而防止了精细规模的密集采样。在随后的作品中，大多数提高NeRF训练和推理速度的尝试可以大致分为以下两类。
- 第一类将NeRF MLP评估结果训练，预计算并存储到更易于访问的数据结构中。这只会提高推理速度，尽管有很大的因素。我们将这些模型称为baked models；
- 第二类是non-baked models。其中包括多种类型的创新。这些模型通常 (但并非总是) 尝试从学习的mlp的参数中学习单独的场景特征，这反过来又允许较小的mlp (例如，在体素网格中学习和存储特征，然后将其馈送到产生颜色和密度的mlp中)，这可以以内存为代价提高训练和推理速度；

&emsp;&emsp;其他技术，例如射线终止 (当累积的透射率接近零时防止进一步的采样点)，空白空间跳过和/或分层采样 (原始NeRF论文中使用的粗细mlp)。这些也经常用于进一步提高训练和推理速度。

### NSVF
&emsp;&emsp;在神经稀疏体素场 (NSVF) (2020年7月) 中，Liu等人。[46] 开发了基于体素的NERF模型，该模型将场景建模为以体素为边界的一组辐射场。通过插值存储在体素顶点的可学习特征来获得特征表示，然后由计算$\sigma$和$c$的共享MLP处理。NSVF对射线使用基于稀疏体素相交的点采样，这比密集采样要有效得多，或Mildenhall等人的分层两步法 [1]。但是，由于将特征向量存储在潜在密集的体素网格上，因此该方法的内存密集型更高。

### AutoInt
&emsp;&emsp;AutoInt (2020年12月) [47] 近似于体积渲染步骤。通过分段分离离散体绘制方程4，然后使用它们新开发的AutoInt，该AutoInt通过训练其梯度 (grad) 网络$\psi_\theta^i$ 来训练MLP $\varPhi_\theta$ ，该网络与内部参数共享，并用于重组积分网络$\varPhi_\theta$ 。这允许渲染步骤使用更少的样本，从而导致比基线NeRF加速十倍的速度略有质量下降。
### DIVeR
&emsp;&emsp;用于体绘制的确定性集成 (DIVeR) [93] (2021年11月) 从NSVF [46] 中汲取了灵感，还在执行稀疏正则化和体素剔除的同时，共同优化了特征体素网格和解码器MLP。但是，他们使用与NeRF方法不同的技术在体积渲染上进行了创新。DIVeR在体素网格上执行了确定性射线采样，该网格为每个射线间隔 (由射线与特定体素的交点定义) 产生了一个集成特征，该特征由MLP解码以产生射线间隔的密度和颜色。这基本上颠倒了体积采样和MLP评估之间的通常顺序。该方法在NeRF合成 [1]，BlendedMVS [94] 以及Tanks和Temple数据集 [95] 上进行了评估，在质量方面优于PlenOctrees [50]，FastNeRF [51] 和KiloNeRF [52] 等方法，具有相当的渲染速度。

### Instant-NGP
&emsp;&emsp;Muller等人最近的一项创新，被称为InstantNeural Graphics原语 (Instant-NGP) [48] (2022年1月)，极大地提高了NERF模型训练和推理速度。作者提出了一种学习的参数多分辨率哈希编码，该编码与NERF模型MLPs同时训练。他们还采用了先进的射线行进技术，包括指数步进，空白空间跳过，样本压缩。MLP的这种新的位置编码和相关的优化实现大大提高了训练和推理速度，以及所得NERF模型的场景重建精度。在训练的几秒钟内，他们取得了与以前NERF模型中数小时训练相似的结果。
### SNeRG
&emsp;&emsp;Yu等人的并行SNeRG，plenotree [50] (3月2021日) 方法。实现了比原始实现快3000倍的推理时间。作者训练了a spherical harmonic NeRF (nerf-sh)，它不是预测颜色函数，而是预测其spherical harmonic 系数。作者建立了预先计算的颜色MLP的spherical harmonic coefﬁcients的八叉树。在八叉树的建造过程中，首先对场景进行了体素化，消除了低透射率体素。通过对NeRF的球谐分量进行蒙特卡洛估计，此过程也可以应用于标准NeRF (non nerf-sh模型)。可以使用初始训练图像进一步优化plenotrees。相对于NeRF训练，这种微调程序是快速的。
### FastNeRF
&emsp;&emsp;在FastNeRF [51] (3月2021日) 中，Garbin等人将颜色函数c分解为与方向位置相关的MLP的输出 (也产生密度 σ) 和与方向相关的MLP的输出的内积。这使Fast-NeRF可以轻松地在场景的密集网格中缓存颜色和密度评估，从而大大提高了推理时间的3000倍。它们还包括硬件加速光线跟踪 [96]，它跳过了空白空间，并在光线透射率饱和时停止。

### KiloNeRF
&emsp;&emsp;Reiser等人 [52] (2021年5月) 通过引入KiloNeRF在基线NeRF上进行了改进，该方法将场景分离为数千个单元格，并训练了独立的mlp以对每个单元格进行颜色和密度预测。这数千个小型MLP是使用来自大型训练有素的老师MLP的知识蒸馏进行培训的，我们发现这与 “烘焙” 密切相关。他们还采用了早期射线终止和空白跳过。仅这两种方法就将基线NeRF的渲染时间提高了71倍。将基线NeRF的MLP分成数千个较小的MLP，进一步将渲染时间提高了36倍，从而使渲染时间加快了2000倍。


## 少量拍摄/稀疏训练视图NeRF
![在这里插入图片描述](https://img-blog.csdnimg.cn/3ef6c10f5a46481e8e043d4203f9cef2.png)
- MVSNeRF (2021):[https://apchenstu.github.io/mvsnerf/](https://apchenstu.github.io/mvsnerf/)*代码已开源*
- NeuRay (2021):[https://liuyuan-pal.github.io/NeuRay/](https://liuyuan-pal.github.io/NeuRay/)*代码已开源*
- PixelNeRF (2020):[https://alexyu.net/pixelnerf/](https://alexyu.net/pixelnerf/)*代码已开源*
- DietNeRF (2021):[https://ajayj.com/dietnerf](https://ajayj.com/dietnerf)*代码已开源*
- DS-NeRF (2021):[https://www.cs.cmu.edu/~dsnerf/](https://www.cs.cmu.edu/~dsnerf/)*代码已开源*

&emsp;&emsp;基线NeRF需要许多具有每个场景已知相机姿势的多视图图像; 每个场景都是独立训练的。基线NeRF的常见失败情况是训练视图变化不够，训练样本不足，或者样本的姿势变化不够。这会导致对单个视图的过度拟合以及不明智的场景几何形状。然而，一系列NeRF方法利用预先训练的图像特征提取网络，通常是预先训练的卷积神经网络 (CNN)，如 [35][100]，大大降低了成功训练NeRF所需的训练样本数量。一些作者称此过程为 “图像特征条件”。与基线NeRF模型相比，这些模型通常具有较低的训练时间。
### pixelNeRF
&emsp;&emsp;在pixelNeRF [58] (2020年12月) 中，Yu等人。使用卷积神经网络 (和双线性插值) 的预训练层来提取图像特征。然后将NeRF中使用的相机射线投影到图像平面上，并为每个查询点提取图像特征。然后将特征，视图方向和查询点传递到NeRF网络上，从而产生密度和颜色。Trevitick等人的一般辐射场 (GRF) [101] (2020年10月) 采取了类似的方法，主要区别在于GRF在规范空间中操作，而不是pixelNeRF的视图空间。

### MVSNeRF
&emsp;&emsp;MVSNeRF [56] (3月2021日) 使用了一种略有不同的方法。他们还使用预先训练的CNN提取了2D图像特征。然后使用平面扫描和基于方差的成本将这些2D特征映射到3D体素化成本体积。预先训练的3D CNN用于提取3D神经编码体积，该体积用于使用插值生成每个点的潜在代码。当为体绘制执行点采样时，NeRF MLP然后使用这些潜在特征，点坐标和观看方向作为输入来生成点密度和颜色。训练过程涉及3D特征卷和NeRF MLP的联合优化。在DTU数据集上进行评估时，在训练的15分钟内，MVSNeRF可以达到与基线NeRF训练数小时相似的结果。

### DietNeRF
&emsp;&emsp;DietNeRF [59] (2021年6月) 除了标准光度损失外，还基于从Clip-ViT [102] 提取的图像特征引入了语义一致性损失$L_{sc}$。

 $$
 L_{sc} = \frac{\lambda}{2}|| \phi(I)-\phi(\hat{I})||_2^2,(21)
 $$
&emsp;&emsp; 其中$\phi$对训练图像I和渲染图像I进行Clip-ViT特征提取。这简化为归一化特征向量的余弦相似性损失 ([59] 中的公式5)。DietNeRF以子采样的NeRF合成数据集 [1] 和DTU数据集 [26] 为基准。单视图新颖合成的最佳性能方法是使用DietNeRF的语义一致性损失进行微调的pixelNeRF [58] 模型。

### NeuRay
&emsp;&emsp;Liu等人 [57] (2021年7月) 的神经射线 (Neural Rays，Neural Rays) 方法也采用了成本量方法。从所有输入视图中，作者使用多视图立体算法估算了成本量 (或深度图)。根据这些，CNN用于创建特征图G。在体绘制过程中，从这些特征中提取可见性和局部特征，并使用MLPs进行处理以提取颜色和alpha。可见性计算为累积密度函数，写为加权和sigmoid函数 $\phi$
 $$
v(z) = 1 − t(z)，where \;t(z) = \sum_{i=1}^N{w_i\varPhi(\frac{(z-\mu_i)}{\sigma_i})}(22)
 $$
&emsp;&emsp;其中$w_i，\mu_i，\sigma_ i$使用MLP从G解码。NeuRay还使用了基于alpha的采样策略，通过计算命中概率，并且仅在具有高命中概率的点周围进行采样 (有关详细信息，请参见 [57] 中的第3.6节)。与其他以从预先训练的神经网络提取的图像特征为条件的NeRF模型一样，NeuRay很好地推广到新场景，并且可以进一步微调以超过基线NeRF模型的10个性能。在对两个模型进行微调30分钟后，NeuRay在NeRF合成数据集上的表现优于MVSNeRF。
### GeoNeRF
&emsp;&emsp;GeoNeRF[103] (2021年11月) 使用预先训练的特征金字塔网络从每个视图中提取2D图像特征。然后，此方法使用平面扫描构建了级联3D成本量。从这两个特征表示中，对于沿射线的N个查询点中的每个，提取了一个视图独立和多个视图依赖的特征标记。这些是使用变压器进行改进的 [104]。然后，通过自动编码器细化与N个视图无关的标记，该自动编码器沿射线返回N个密度。N组与视图相关的令牌分别输入到提取颜色的MLP中。如作者所示，这些网络都可以经过预先训练并很好地推广到新场景。此外，它们可以对每个场景进行微调，在DTU [26]，NeRF合成 [1] 和LLF前向 [5] 数据集上取得很好的效果，优于pixelNeRF [58] 和MVSNeRF [56] 等方法。
## (潜在) 条件NeRF
![在这里插入图片描述](https://img-blog.csdnimg.cn/ff40ac1b5d09473db5a66438f7b829b7.png)
- GIRAFFE (2020):[https://m-niemeyer.github.io/project-pages/giraffe/index.html](https://m-niemeyer.github.io/project-pages/giraffe/index.html)*代码已开源*
- GRAF (2020):[https://autonomousvision.github.io/graf/](https://autonomousvision.github.io/graf/)*代码已开源*
- $\pi$-GAN (2020):[https://marcoamonteiro.github.io/pi-GAN-website/](https://marcoamonteiro.github.io/pi-GAN-website/)*代码已开源*
- GNeRF (2021):[https://github.com/quan-meng/gnerf](https://github.com/quan-meng/gnerf)*代码已开源*
- NeRF-VAE (2021):[https://arxiv.org/abs/2104.00587](https://arxiv.org/abs/2104.00587)
- NeRF-W (2020):[https://nerf-w.github.io/](https://nerf-w.github.io/)(非官方实现代码:[https://github.com/kwea123/nerf_pl](https://github.com/kwea123/nerf_pl))
- Edit-NeRF (2021):[http://editnerf.csail.mit.edu/](http://editnerf.csail.mit.edu/)*代码已开源*
- CLIP-NeRF (2021):[https://arxiv.org/abs/2112.05139](https://arxiv.org/abs/2112.05139)

&emsp;&emsp;NeRF模型的潜在条件是指使用潜在向量 (又名潜在代码) 来控制NeRF视图合成的各个方面。这些潜在向量可以在管道的各个点处输入，以控制场景的组成，形状和外观。它们允许添加一组参数来控制改变图像到图像的场景方面，同时允许其他部分考虑场景的永久方面，例如场景几何形状。训练以潜码为条件的图像生成器的一种相当简单的方法是使用变分自动编码器 (VAE) [112] 方法。这些模型使用编码器和解码器，编码器按照用户定义的特定概率分布将图像转换为潜码，解码器将采样的潜码转换回图像。与以下两种方法相比，这些方法在NeRF模型中使用的频率不高，因此，我们不为VAE引入单独的小节。
## 解绑场景与场景构图
![在这里插入图片描述](https://img-blog.csdnimg.cn/c504e4663c454e2f93c09704adc0f0af.png)
- NeRF-W (2020):[https://nerf-w.github.io/](https://nerf-w.github.io/)(非官方实现代码:[https://github.com/kwea123/nerf_pl](https://github.com/kwea123/nerf_pl))
- NeRF++ (2020):[https://github.com/Kai-46/nerfplusplus](https://github.com/Kai-46/nerfplusplus)*代码已开源*
- GIRAFFE (2020):[https://m-niemeyer.github.io/project-pages/giraffe/index.html](https://m-niemeyer.github.io/project-pages/giraffe/index.html)*代码已开源*
- Fig-NeRF (2021):[https://fig-nerf.github.io/](https://fig-nerf.github.io/)
- Object-NeRF (2021):[https://zju3dv.github.io/object_nerf/](https://zju3dv.github.io/object_nerf/)
Semantic-NeRF (2021):[https://shuaifengzhi.com/Semantic-NeRF/](https://shuaifengzhi.com/Semantic-NeRF/)*代码已开源*

&emsp;&emsp;尝试将NeRF模型应用于户外场景时，需要将前景和背景分开。这些室外场景还在照明和外观的图像图像变化方面提出了额外的挑战。本节介绍的模型使用各种方法解决了这个问题，许多模型通过逐个图像的外观代码来适应潜在条件。此研究领域中的某些模型还执行语义或实例分割，以查找3D语义标记中的应用。
### NeRF-W
&emsp;&emsp;在NeRF In the Wild (nerf-w) [65] (2020年8月) 中，Martin-Brualla等人解决了基线NeRF模型的两个关键问题。同一场景的真实照片可以包含由于照明条件而导致的每幅图像外观变化，以及在每幅图像中不同的瞬态对象。对于场景中的所有图像，密度MLP保持固定。但是，nerf-w将其颜色MLP取决于每个图像的外观嵌入。此外，另一个以每个图像瞬态嵌入为条件的MLP预测了瞬态对象的颜色和密度函数。这些潜在嵌入是使用生成潜在优化构建的。Nerf-w在渲染速度方面没有改善NeRF，但在拥挤的光旅游数据集上取得了更高的结果 [121]。
### NeRF++
&emsp;&emsp;Zhang等。开发了NeRF [67] (2020年10月) 模型，该模型通过使用球体分离场景而适应于生成未绑定场景的新颖视图。球体的内部包含所有前景对象和所有虚拟摄像机视图，而背景则位于球体之外。然后使用倒置的球体空间重新参数化球体的外部。训练了两个单独的NeRF模型，一个用于球体内部，另一个用于外部。相机射线积分也分为两部分进行了评估。使用这种方法，他们在坦克和太阳穴 [95] 场景以及来自yuger等人 [122] 的场景上的表现优于基线NeRF。
### GIRAFFE
&emsp;&emsp;GIRAFFE [60] (2020年11月) 也采用与nerf-w类似的方法构建，使用生成潜在代码并将背景和前景MLP分离以进行场景构图。GIRAFFE 是基于格拉夫的。它为场景中的每个对象分配了MLP，该MLP产生了标量密度和深层特征向量 (替换颜色)。这些mlp (具有共享的体系结构和权重) 作为输入形状和外观潜在向量以及输入姿势。背景被视为所有其他对象，除了它自己的MLP和权重。然后使用特征的密度加权和组成场景。然后使用体绘制从该3D体积特征字段创建一个小的2D特征图，该图被馈送到上采样CNN中以生成图像。GIRAFFE 使用此合成图像和2D CNN鉴别器进行了对抗性训练。生成的模型具有分散的潜在空间，可以对场景生成进行精细控制。
### Fig-NeRF
&emsp;&emsp;Fig-NeRF [68] (2021年4月) 也采用场景构图，但侧重于对象插值和偶数分割。使用两个单独的NeRF模型，一个用于前景，一个用于背景。他们的前景模型是可变形的Nerfies模型 [39]。他们的背景模型是外观潜码条件的NeRF。他们使用了两个光度损失，一个用于前景，一个用于背景。Fig-NeRF在诸如ShapeNet [30] Gelato [123] 和Objectron [124] 之类的数据集上获得了良好的结果。
## 姿态估计
![在这里插入图片描述](https://img-blog.csdnimg.cn/ea47af15b8d346a6bdd1b3e3f4abda91.png)
- iMAP (2021):[https://en.wikipedia.org/wiki/Internet_Message_Access_Protocol](https://en.wikipedia.org/wiki/Internet_Message_Access_Protocol)
- NICE-SLAM (2021):[https://pengsongyou.github.io/nice-slam](https://pengsongyou.github.io/nice-slam)*代码已开源*
- NeRF-- (2020):[https://nerfmm.active.vision/](https://nerfmm.active.vision/)*代码已开源*
- BARF (2021):[https://chenhsuanlin.bitbucket.io/bundle-adjusting-NeRF/](https://chenhsuanlin.bitbucket.io/bundle-adjusting-NeRF/)*代码已开源*
- SCNeRF (2021):[https://postech-cvlab.github.io/SCNeRF/](https://postech-cvlab.github.io/SCNeRF/)*代码已开源*


&emsp;&emsp;NeRF模型需要输入图像和相机姿势来训练。在最初的2020论文中，未知的姿势被COLMAP库 [2] 获取，在没有提供相机姿势的情况下，该库也经常用于许多后续的NeRF模型中。通常，通过运动 (SfM) 问题将同时执行姿势估计和使用NeRF隐式场景表示的构建模型公式化为离线结构。在这些情况下，通常使用束调整 (BA) 来共同优化姿势和模型。但是，某些方法也将其表述为在线同时定位和映射 (SLAM) 问题。
### iNeRF
&emsp;&emsp;iNeRF [125] (2020年12月) 将姿势重建公式化为一个反问题。给定预先训练的NeRF，使用光度尺损失8，Yen-Chen等人。优化了姿势而不是网络参数。作者使用了兴趣点检测器，并进行了基于兴趣区域的采样。作者还进行了半监督实验，在该实验中，他们对未摆姿势训练图像使用iNeRF姿势估计来增强NeRF训练集，并进一步训练前向NeRF。作者表示，这种半监督将前向NeRF提出的照片的要求降低了25%。

### NeRF--
&emsp;&emsp;NeRF- [73] (2021年2月) 共同估计了NeRF模型参数和相机参数。这允许模型以端到端的方式构造辐射场并仅合成新颖的视图图像。NeRF-就两个视图合成而言，总体上获得了与使用COLMAP和2020 NeRF模型相当的结果。但是，由于姿势初始化的限制，NeRF-最适合前置场景，并且在旋转运动和对象跟踪运动方面苦苦挣扎。

### BARF
&emsp;&emsp;与NeRF-同时出现的是束调整的神经辐射场 (BARF) [74] (2021年4月)，它还与神经辐射场的训练一起共同估计了姿势。BARF还通过自适应掩蔽位置编码来使用从粗到细的配准，类似于Nerfies中使用的技术 [39]。总体而言，BARF结果超过了NeRF-在具有未知相机姿势的LLFF前向场景数据集上，1.49 PNSR在八个场景中的平均值，并超过了COLMAP注册基线NeRF的0.45 PNSR。为了简单起见，BARF和NeRF都使用了朴素的密集射线采样。
### SCNeRF
&emsp;&emsp;Jeong等人介绍了一种用于NeRF的自校准联合优化模型 (SCNeRF) [75] (2021年8月)。他们的相机校准模型不仅可以优化未知姿势，而且可以优化非线性相机模型 (例如鱼眼镜头模型) 的相机固有。通过使用课程学习，他们逐渐将非线性相机/噪声参数引入联合优化中。该相机优化模型也是模块化的，可以轻松地与不同的NeRF模型一起使用。该方法在LLFF场景 [5] 上优于BARF [74]。

### IMAP
&emsp;&emsp;Sucar等人推出了第一个基于NeRF的密集在线SLAM模型，名为iMAP [71] (3月2021日)。该模型利用持续的在线学习，以NeRF模型的形式共同优化了相机姿势和隐式场景表示。他们使用了迭代的两步方法跟踪 (相对于NeRF的姿势优化) 和映射 (姿势和NeRF模型参数的束调整联合优化)。iMAP通过与映射过程并行运行更快的跟踪步骤，实现了接近相机帧速率的姿势跟踪速度。iMAP还通过在稀疏和增量选择的一组图像上优化场景来使用关键帧选择。
### GNeRF
&emsp;&emsp;GNeRF是Meng等人 [63] (3月2021日) 的一种不同类型的方法，使用pose作为生成潜码。GNeRF首先通过对抗性训练获得粗略的相机姿势和辐射场。这是通过使用生成器来完成的，该生成器采用随机采样的姿势，并使用NeRF风格的rending合成视图。然后，鉴别器将渲染的视图与训练图像进行比较。然后，反转网络获取生成的图像，并输出一个姿势，将其与采样的姿势进行比较。这导致了粗略的图像姿势配对。然后在混合优化方案中通过光度损失共同完善图像和姿势。在合成NeRF数据集上，基于COLMAP的NeRF略优于GNeRF，在DTU数据集上优于基于COLMAP的NeRF。

### NICE-SLAM
&emsp;&emsp;在iMAP的基础上，NICE-SLAM [72] (2021年12月) 在关键帧选择和NeRF体系结构等各个方面进行了改进。具体来说，他们使用了场景几何形状的基于分层网格的表示，该表示能够在某些场景的大规模未观察到的场景特征 (墙壁，地板等) 中填补iMAP重建的空白。NICESLAM比iMAP实现了更低的姿态估计误差和更好的场景重建结果。NICE-SLAM也只使用iMAP的1/4触发器，1/3跟踪时间和1/2映射时间。
## 相邻方法
### Plenoxel
&emsp;&emsp;Plenoxel [53] (2021年12月) 跟随plenotree的脚步，对场景进行了体素化，并存储了密度和球谐系数与方向相关的颜色的标量。然而，令人惊讶的是，Plenoxel完全跳过了MLP训练，而是将这些功能直接安装在体素网格上。他们还获得了与NeRF和JaxNeRF相当的结果，训练时间加快了几百倍。这些结果表明，NeRF模型的主要贡献是给定密度和颜色的新视图的体积渲染，而不是密度和颜色mlp本身。
### TensorRF
&emsp;&emsp;沿着无MLP辐射场的路线，TensorRF [55] (3月2022日) 也使用了无神经网络的体绘制。TensorRF在3D体素网格中存储了标量密度和矢量颜色特征 (SH谐波系数或要输入到小MLP中的颜色特征)，然后将其表示为秩3张量$T_\sigma∈ R^{H × W × D}$和秩4张量 $T_c∈ R^{H × W × D × C}$，其中H，W，D是体素网格的高度，宽度和深度分辨率，C是通道尺寸。然后，作者使用了两种因式分解方案: 将张量分解为纯矢量外积的candecompp-PARAFAC (CP) 和将张量分解为矢量/矩阵外积的矢量矩阵 (VM)。当使用CP时，这些分解将来自Plenoxels的内存需求降低了200倍。尽管需要更多内存，但它们的VM分解效果最佳。训练速度与Pleoxels相当，并且比基准NeRF模型快得多。
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

$$L_{sight}=E\left[ \int_{_{t_1}}^{t_2}{\left( w\left( t \right) −\delta \left( z \right) \right)^2dt} \right] ,(31)$$

$$L_{seg}=E\left[ S_i\left(r\int_{_{t_1}}^{t_2}{\left( w\left( t \right) −\delta \left( z \right) \right)^2dt} \right)\right] ,(32)$$

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

$$\sigma \left( x \right) =M_\sigma \left( \varphi \left( x|Z，S \right) \right) ,(33)$$

$$c \left( x \right) =M_\sigma \left( \varphi \left( x|Z，S \right)  ,γ_x(x),γ_d(d),lt\right),(34)$$

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
$$L=\sum_{i=1}^N{\left( \frac{\hat{y}_i-y_i}{sg\left( \hat{y}_i \right) +\epsilon} \right)^2},(35)$$
&emsp;&emsp;其中sg(.) 表示梯度停止 (将其参数视为具有零梯度的常数)。RawNeRF由可变曝光图像监督，NeRF模型的 “曝光” 由训练图像的快门速度以及每通道学习的缩放因子缩放。它在夜间和弱光场景渲染和去噪方面取得了令人印象深刻的效果。RawNeRF特别适合光线不足的场景。关于基于NeRF的去噪，NaN [135] (4月10日) 也探索了这一新兴的研究领域。
#### HDR-NeRF
&emsp;&emsp;与RawNeRF同时，Xin等人的HDR-NeRF [133] (2021年11月) 也致力于HDR视图合成。但是，与RawNeRF中的原始线性图像相反，HDR-NeRF通过使用具有可变曝光时间的低动态范围训练图像来进行HDR视图合成。RawNeRF建模了HDR辐射e(r) ∈ [0，∞) 取代了 (1) 中的标准c(r)。使用三个MLP摄像机响应函数 (每个颜色通道一个) f，该辐射被映射到彩色c。这些代表了典型的摄像机制造商依赖的线性和非线性后处理。HDR-NeRF的性能大大优于基线NeRF和nerf-w [65] 在低动态范围 (LDR) 重建中，并在HDR重建方面获得了较高的视觉评估得分。

#### Semantic-NeRF
&emsp;&emsp;Semantic-NeRF [70] (2021年3月) 是一个能够合成新颖视图的语义标签的NeRF模型。这是使用附加的与方向无关的MLP (分支) 完成的，该MLP将位置和密度MLP特征作为输入，并生成逐点语义标签$s$。语义标签也是通过体绘制生成的。
$$S(r) = \sum _i^N{T_i\alpha_is_i},(36)$$
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

$$\rho \left( t \right) =max\left( \frac{\frac{−d\varPhi}{dt}\left( f\left( r\left( t \right) \right) \right)}{ \varPhi \left( f\left( r\left( t \right) \right) \right)} ,0 \right) ,(37)$$

&emsp;&emsp;其中 $\varPhi(·)$是$s$形函数，其导数$\frac{d\varPhi}{dt}$是logistic密度分布。作者已经证明了他们的模型优于基线NeRF，并为他们的方法和基于SDF的场景密度的实现提供了理论和实验依据。此方法与UNISERF并发，并且在DTU数据集上优于它 [26]。与UNISURF一样，可以使用在SDF上执行根查找来定义场景的显式表面几何形状。
#### Others
&emsp;&emsp;Azinovic等人的同时工作[137] (2021年4月) 也用截断的SDF MLP代替了密度MLP。相反，他们将像素颜色计算为采样颜色的加权总和

$$C\left( r \right) =\frac{\sum_{i=1}^N{w_ic_i}}{\sum_{i=1}^N{w_i}},(38)$$

其中$w_i$由$s$形函数的乘积给出，$s$形函数由

$$w_i = \varPhi(\frac{D_i}{tr}) · \varPhi(-\frac{D_i}{tr})	,(39)$$

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

