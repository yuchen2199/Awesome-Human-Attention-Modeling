<h1 align="center">
  Awesome – Human Attention Modeling & Applications
</h1>

<p align="center">
  <a href="https://awesome.re">
    <img src="https://awesome.re/badge.svg" alt="Awesome">
  </a>
</p>

<p align="center">
  A curated collection of papers, datasets, benchmarks, code, and applications
  on human visual attention, including saliency prediction, scanpath prediction,
  and attention-aware applications.
</p>

---

## 📣 Latest News

> ❗ **Latest update:** 22 November 2025  
> ❗ **Status:** This repository is a work in progress. New updates coming soon, stay tuned!! 🎉  

---

## 📌 Introduction

Human visual attention plays a central role in how we perceive, filter, and act on information.  
It selectively allocates limited cognitive resources, shapes visual perception, and drives behavior
in tasks such as search, recognition, navigation, and interaction.

In modern AI, especially computer vision and multimodal learning, there is a growing effort to
**model and leverage human-like attention**. Neural attention mechanisms, eye-tracking–based supervision,
and human-in-the-loop evaluation all aim to:

- make models focus on task-relevant visual evidence,
- improve robustness and generalization across domains and modalities,
- enhance interpretability and human alignment.

Understanding and modeling human visual attention is therefore crucial for:

- **Cross-modal learning** (e.g., vision–language models grounded in human gaze or saliency),
- **Multi-task and interactive systems** (e.g., agents that coordinate perception, action, and communication),
- **Human-centered applications** (e.g., HCI, AR/VR, medical imaging, assistive systems).

This repository is dedicated to:

- 📚 Summarizing **recent papers and advances** in human visual attention modeling  
- 🧠 Bridging **human attention and AI attention mechanisms** in vision and multimodal models  
- 🧪 Providing **datasets, code links, and benchmarks** for saliency and scanpath prediction  
- 🧷 Highlighting **applications** that explicitly leverage human visual attention signals  

---

## 📂 Main Topics

- 📄 [Saliency Prediction](#saliency-prediction)
- 📄 [Scanpath Prediction](#scanpath-prediction)
- 📄 [Applications of Human Visual Attention](#applications-of-human-visual-attention)
- 📄 [Resources & Benchmarks](#resources-and-benchmarks)

---

## 📕 Table of Contents

- 🌸 [Saliency Prediction](#saliency-prediction)
  - [Datasets](#saliency-prediction-datasets)
  - [Methods](#saliency-prediction-methods)
    - [Image](#saliency-image-methods)
    - [3D](#saliency-3d-methods)
    - [Video](#saliency-video-methods)
    - [Other](#saliency-other-methods)

- 🌊 [Scanpath Prediction](#scanpath-prediction)
  - [Datasets](#scanpath-prediction-datasets)
  - [Methods](#scanpath-prediction-methods)
    - [Image](#scanpath-image-methods)
    - [3D](#scanpath-3d-methods)
    - [Video](#scanpath-video-methods)
    - [Other](#scanpath-other-methods)

- 🎯 [Applications of Human Attention](#applications-of-human-visual-attention)
  - [Datasets](#applications-datasets)
  - [Methods](#applications-methods)
    - [HCI / Interaction / AR](#applications-hci)
    - [AD / Cognitive & Neuro](#applications-ad)
    - [Robotics](#applications-robotics)
    - [Medicine](#applications-medicine)
    - [Design](#applications-design)
    - [Commerce](#applications-commerce)

- 📑 [Resources & Benchmarks](#resources-and-benchmarks)


---
<a id="saliency-prediction"></a>
## 🌸 Saliency Prediction


### 1. Datasets 

<details open id="saliency-prediction-datasets">
<summary>Dataset list</summary>
<br>

| Conference / Journal | Title | Links |
|----------------------|-------|-------|
| ECCV 2024 | [Early Anticipation of Driving Maneuvers](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08862.pdf) | [Dataset / Project](https://cvit.iiit.ac.in/research/projects/cvit-projects/daad#dataset) |
| IJCV 2024 | [Joint Learning of Audio–Visual Saliency Prediction and Sound Source Localization on Multi-face Videos](https://link.springer.com/article/10.1007/s11263-023-01950-3) |  |
| TCSVT 2024 | [Saliency Prediction on Mobile Videos: A Fixation Mapping-Based Dataset and A Transformer Approach](https://ieeexplore.ieee.org/abstract/document/10360106) | [GitHub](https://github.com/wenshijie110/MVFormer) |
| TITS 2024 | [Repeated Route Naturalistic Driver Behavior Analysis Using Motion and Gaze Measurements](https://doi.org/10.1109/TITS.2024.3520893) | [GitHub](https://bikram11.github.io/Website_R2ND2/) |
| TIV 2024 | [Data Limitations for Modeling Top-Down Effects on Drivers' Attention](https://doi.org/10.1109/IV55156.2024.10588528) | [GitHub](https://github.com/ykotseruba/SCOUT) |
| CHI 2023 | [UEyes: Understanding Visual Saliency Across User Interface Types](https://dl.acm.org/doi/pdf/10.1145/3544548.3581096) | [GitHub](https://github.com/YueJiang-nj/UEyes-CHI2023) |
| TITS 2023 | [Driving Visual Saliency Prediction of Dynamic Night Scenes via a Spatio-Temporal Dual-Encoder Network](https://doi.org/10.1109/TITS.2023.3323468) | [GitHub](https://github.com/taodeng/DrFixD-night) |
| CVPR 2022 | [Does Text Attract Attention on E-Commerce Images: A Novel Saliency Prediction Dataset and Method](https://openaccess.thecvf.com/content/CVPR2022/papers/Jiang_Does_Text_Attract_Attention_on_E-Commerce_Images_A_Novel_Saliency_CVPR_2022_paper.pdf) | [Project](https://openaccess.thecvf.com/content/CVPR2022/papers/Jiang_Does_Text_Attract_Attention_on_E-Commerce_Images_A_Novel_Saliency_CVPR_2022_paper.pdf) |
| ECCV 2022 | [Look Both Ways: Self-Supervising Driver Gaze Estimation and Road Scene Saliency](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/1367.pdf) | [GitHub](https://github.com/Kasai2020/look_both_ways) |
| CAA 2022 | [Driving as Well as on a Sunny Day? Predicting Driver's Fixation in Rainy Weather Conditions](https://doi.org/10.1109/JAS.2022.105716) | [GitHub](https://github.com/taodeng/DrFixD-rainy) |
| ICCVW 2021 | [MAAD: A Model and Dataset for “Attended Awareness” in Driving](https://openaccess.thecvf.com/content/ICCV2021W/EPIC/papers/Gopinath_MAAD_A_Model_and_Dataset_for_Attended_Awareness_in_Driving_ICCVW_2021_paper.pdf) | [GitHub](https://github.com/ToyotaResearchInstitute/att-aware/) |
| ECCV 2020 | [MVVA: A Large-Scale Multi-Modality Visual-Audio Saliency Dataset](https://arxiv.org/abs/2103.15438) | [Dataset](https://github.com/MinglangQiao/MVVA-Database) / [GitHub](https://github.com/MinglangQiao/visual_audio_saliency) |
| ITSC 2019 | [DADA-2000: Can Driving Accident Be Predicted by Driver Attention?](https://doi.org/10.1109/ITSC.2019.8917218) | [GitHub](https://github.com/JWFangit/LOTVS-DADA) |
| TITS 2019 | [How Do Drivers Allocate Their Potential Attention? Driving Fixation Prediction via CNNs](https://doi.org/10.1109/TITS.2019.2915540) | [GitHub](https://github.com/taodeng/CDNN-traffic-saliency) |
| TIV 2019 | [How to Evaluate Object-of-Fixation Detection](https://doi.org/10.1109/IVS.2019.8814224) | [Dataset](https://www.proreta.tu-darmstadt.de/proreta_1_4/proreta4_1/datasets_1/index.en.jsp) |
| TVCG 2018 | [Saliency in VR: How Do People Explore Virtual Environments?](https://ieeexplore.ieee.org/abstract/document/8269807) | [GitHub](https://github.com/vsitzmann/vr-saliency) |
| TPAMI 2018 | [Personalized Saliency and Its Prediction](https://ieeexplore.ieee.org/abstract/document/8444709) | [GitHub](https://github.com/xuyanyu-shh/Personalized-Saliency) |
| TPAMI 2018 | [The DR(eye)VE Project: Predicting the Driver's Focus of Attention](https://doi.org/10.1109/TPAMI.2018.2845370) | [Dataset](http://imagelab.ing.unimore.it/dreyeve) |
| ACCV 2018 | [Predicting Driver Attention in Critical Situations](https://doi.org/10.1007/978-3-030-20873-8_42) | [Dataset](https://bdd-data.berkeley.edu/) |
| ECCV 2018 | [LEDOV: Large-Scale Eye-Tracking Database for Visual Saliency in Videos](https://arxiv.org/pdf/1709.06316v3) | [Dataset](https://github.com/remega/LEDOV-eye-tracking-database) |
| ECCV 2018 | [Task-Driven Webpage Saliency](https://openaccess.thecvf.com/content_ECCV_2018/html/Quanlong_Zheng_Task-driven_Webpage_Saliency_ECCV_2018_paper.html) | [GitHub](https://github.com/quanlzheng/Task-driven-Webpage-Saliency) / [Project](https://quanlzheng.github.io/projects/Task-driven-Webpage-Saliency.html) |
| CVPR 2018 | [DHF1K: A Large-Scale Benchmark for Video Saliency](https://arxiv.org/abs/1801.07424) | [GitHub](https://github.com/wenguanwang/DHF1K) |
| CVPR 2018 | [Cube Padding for Weakly-Supervised Saliency Prediction in 360° Videos](https://arxiv.org/abs/1806.01320v1) | [GitHub](https://github.com/hsientzucheng/CP-360-Weakly-Supervised-Saliency) |
| IJCAI 2017 | [Beyond Universal Saliency: Personalized Saliency Prediction with Multi-task CNN](https://www.ijcai.org/proceedings/2017/0543.pdf) | [GitHub](https://github.com/xuyanyu-shh/Personalized-Saliency) |
| TITS 2016 | [Where Does the Driver Look? Top-Down-Based Saliency Detection in a Traffic Driving Environment](https://doi.org/10.1109/TITS.2016.2535402) | [GitHub](https://github.com/taodeng/traffic-eye-tracking-dataset) |
| Vision Research 2015 | [Intrinsic and Extrinsic Effects on Image Memorability](https://www.sciencedirect.com/science/article/pii/S0042698915000930) | [Dataset](http://figrim.mit.edu/) |
| arXiv 2015 | [CAT2000: A Large Scale Fixation Dataset for Boosting Saliency Research](https://arxiv.org/pdf/1505.03581) | [Dataset](http://saliency.mit.edu/) |
| CVPR 2015 | [SALICON: Saliency in Context](https://openaccess.thecvf.com/content_cvpr_2015/papers/Jiang_SALICON_Saliency_in_2015_CVPR_paper.pdf) |  |
| JoV 2014 | [Predicting Human Gaze Beyond Pixels](https://jov.arvojournals.org/article.aspx?articleid=2193943) | [GitHub](https://github.com/NUS-VIP/predicting-human-gaze-beyond-pixels) |
| ECCV 2014 | [Saliency in Crowd](https://link.springer.com/chapter/10.1007/978-3-319-10584-0_2) |  |
| CVPR 2014 | [The Secrets of Salient Object Segmentation](https://openaccess.thecvf.com/content_cvpr_2014/html/Li_The_Secrets_of_2014_CVPR_paper.html) |  |
| CVPR 2013 | [Saliency Detection via Graph-Based Manifold Ranking](https://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Yang_Saliency_Detection_via_2013_CVPR_paper.pdf) |  |
| Cognitive Computing 2011 | [Predicting Eye Fixations on Complex Visual Stimuli Using Local Symmetry](https://link.springer.com/article/10.1007/s12559-010-9089-5) |  |
| BMVC 2011 | [Computational Modeling of Top-Down Visual Attention in Interactive Environments](http://www.bmva.org/bmvc/2011/proceedings/paper85/paper85.pdf) | [GitHub](http://ilab.usc.edu/borji/Resources.html) |
| ECCV 2010 | [An Eye Fixation Database for Saliency Detection in Images](https://link.springer.com/chapter/10.1007/978-3-642-15561-1_3) |  |
| JoV 2009 | Saliency, Attention, and Visual Search: An Information Theoretic Approach |  |
| ICCV 2009 | [Learning to Predict Where Humans Look](https://ieeexplore.ieee.org/abstract/document/5459462) | [Dataset](https://people.csail.mit.edu/tjudd/WherePeopleLook/index.html) |

</details>

---
<a id="saliency-prediction-methods"></a>
### 2. Methods 

<details open id="saliency-image-methods">
<summary>Image</summary>
<br>

| Conference / Journal | Title | Links |
|----------------------|-------|-------|
| CVPR 2025 | [Explainable Saliency: Articulating Reasoning with Contextual Prioritization](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Explainable_Saliency_Articulating_Reasoning_with_Contextual_Prioritization_CVPR_2025_paper.html) | [GitHub](https://github.com/NuoChen1203/Explainable_Saliency) |
| AAAI 2025 | [SalM²: An Extremely Lightweight Saliency Mamba Model for Real-Time Cognitive Awareness of Driver Attention](https://doi.org/10.1609/aaai.v39i2.32157) | [GitHub](https://github.com/zhao-chunyu/SaliencyMamba) |
| WACV 2025 | [SUM: Saliency Unification through Mamba for Visual Attention Modeling](https://arxiv.org/pdf/2406.17815) | [GitHub](https://github.com/Arhosseini77/SUM) |
| ECCV 2024 | [Data Augmentation via Latent Diffusion for Saliency Prediction](https://link.springer.com/chapter/10.1007/978-3-031-73229-4_21) | [GitHub](https://github.com/IVRL/Augsal) |
| WACV 2024 | [Learning Saliency From Fixations (SalTR)](https://arxiv.org/pdf/2311.14073) | [GitHub](https://github.com/YasserdahouML/SalTR) |
| TCSVT 2024 | [Quality Assessment and Distortion-aware Saliency Prediction for AI-Generated Omnidirectional Images](https://ieeexplore.ieee.org/abstract/document/11185165) | [GitHub](https://github.com/IntMeGroup/AIGCOIQA) |
| CVPR 2023 | [TempSAL: Uncovering Temporal Information for Deep Saliency Prediction](https://arxiv.org/abs/2301.02315) | [GitHub](https://github.com/IVRL/Tempsal) |
| Neurocomputing 2023 | [TranSalNet: Towards Perceptually Relevant Visual Saliency Prediction](https://www.sciencedirect.com/science/article/pii/S0925231222004714) | [GitHub](https://github.com/LJOVO/TranSalNet) |
| CVPR 2023 | [Learning from Unique Perspectives: User-aware Saliency Modeling](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Learning_From_Unique_Perspectives_User-Aware_Saliency_Modeling_CVPR_2023_paper.pdf) | – |
| CVPR 2020 | [How Much Time Do You Have? Modeling Multi-Duration Saliency](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fosco_How_Much_Time_Do_You_Have_Modeling_Multi-Duration_Saliency_CVPR_2020_paper.pdf) | [GitHub](https://github.com/diviz-mit/multiduration-saliency) |
| ICPR 2020 | [FastSal: A Computationally Efficient Network for Visual Saliency Prediction](https://ieeexplore.ieee.org/abstract/document/9413057) | [GitHub](https://github.com/feiyanhu/FastSal) |
| Image and Vision Computing 2020 | [EML-NET: An Expandable Multi-Layer Network for Saliency Prediction](https://www.sciencedirect.com/science/article/pii/S0262885620300196) | [GitHub](https://github.com/SenJia/EML-NET-Saliency) |
| NeurIPS 2019 | [DeepUSPS: Deep Robust Unsupervised Saliency Prediction via Self-supervision](https://proceedings.neurips.cc/paper/2019/hash/54229abfcfa5649e7003b83dd4755294-Abstract.html) | [GitHub](https://github.com/donnydonnyullrich/DeepUSPS) |
| CVPR 2018 | [SAM: Pushing the Limits of Saliency Prediction Models](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w36/html/Cornia_SAM_Pushing_the_CVPR_2018_paper.html) | [GitHub](https://github.com/ml-lab/sam) |
| CVIU 2018 | [SalGAN: Visual Saliency Prediction with Adversarial Networks](https://arxiv.org/pdf/1701.01081) | [GitHub](https://imatge-upc.github.io/saliency-salgan-2017/) |
| TIP 2018 | [Deep Visual Attention Prediction (DVA)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8240654) | [GitHub](https://github.com/wenguanwang/deepattention) |
| TIP 2017 | [DeepFix: A Fully Convolutional Neural Network for Predicting Human Eye Fixations](https://arxiv.org/pdf/1510.02927) | – |
| CVPR 2016 | [Shallow and Deep Convolutional Networks for Saliency Prediction (SalNet)](https://openaccess.thecvf.com/content_cvpr_2016/papers/Pan_Shallow_and_Deep_CVPR_2016_paper.pdf) | [GitHub](https://github.com/imatge-upc/saliency-2016-cvpr) |
| ICPR 2016 | [A Deep Multi-level Network for Saliency Prediction (MLNet)](https://ieeexplore.ieee.org/abstract/document/7900174) | [GitHub](https://github.com/marcellacornia/mlnet) |
| CVPR 2014 | [Large-Scale Optimization of Hierarchical Features for Saliency Prediction in Natural Images (eDN)](https://openaccess.thecvf.com/content_cvpr_2014/html/Vig_Large-Scale_Optimization_of_2014_CVPR_paper.html) | [GitHub](https://github.com/coxlab/edn-cvpr2014) |
| ICLR 2014 | [Deep Gaze I: Boosting Saliency Prediction with Feature Maps Trained on ImageNet](https://arxiv.org/abs/1411.1045) | [GitHub](https://github.com/matthias-k/DeepGaze) |
| ICCV 2013 | [Saliency Detection: A Boolean Map Approach (BMS)](https://www.cv-foundation.org/openaccess/content_iccv_2013/html/Zhang_Saliency_Detection_A_2013_ICCV_paper.html) | [GitHub](https://github.com/fzliu/saliency-bms) |
| ICCV 2009 | [Learning to Predict Where Humans Look](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5459462) | [Dataset / Project](https://people.csail.mit.edu/tjudd/WherePeopleLook/index.html) |
| Journal of Vision 2008 | [SUN: A Bayesian Framework for Saliency Using Natural Statistics](https://jov.arvojournals.org/article.aspx?articleid=2297284) | – |
| NeurIPS 2006 | [Graph-based Visual Saliency (GBVS)](https://proceedings.neurips.cc/paper_files/paper/2006/hash/4db0f8b0fc895da263fd77fc8aecabe4-Abstract.html) | [GitHub](https://github.com/shreelock/gbvs) |
| NeurIPS 2005 | [Saliency Based on Information Maximization (AIM)](https://proceedings.neurips.cc/paper/2005/hash/0738069b244a1c43c83112b735140a16-Abstract.html) | [GitHub](https://github.com/TsotsosLab/AIM) |
| Vision Research 2000 | [A Saliency-based Search Mechanism for Overt and Covert Shifts of Visual Attention (ITTI)](https://www.sciencedirect.com/science/article/pii/S0042698999001637) | – |
| TPAMI 1998 | [A Model of Saliency-Based Visual Attention for Rapid Scene Analysis (NVT)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=730558) | – |

</details>


<details open id="saliency-3d-methods">
<summary>3D</summary>
<br>

| Conference / Journal | Title | Links |
|----------------------|-------|-------|
| TVCG 2025 | [Unified Approach to Mesh Saliency: Evaluating Textured and Non-Textured Meshes Through VR and Multifunctional Prediction](https://ieeexplore.ieee.org/abstract/document/10918849) |  |
| CVPR 2025 | [Mesh Mamba: A Unified State Space Model for Saliency Prediction in Non-Textured and Textured Meshes](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Mesh_Mamba_A_Unified_State_Space_Model_for_Saliency_Prediction_CVPR_2025_paper.html) | [GitHub](https://github.com/kaviezhang/MeshMamba) |
| TMM 2024 | [Towards 3D Colored Mesh Saliency: Database and Benchmarks](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10251829) |  |
| TMM 2023 | [SP-Det: Leveraging Saliency Prediction for Voxel-Based 3D Object Detection in Sparse Point Cloud](https://ieeexplore.ieee.org/abstract/document/10214314) |  |
| TPAMI 2023 | 3D Visual Saliency: An Independent Perceptual Measure or a Derivative of 2D Image Saliency? |  |
| CVPR 2021 | [Mesh Saliency: An Independent Perceptual Measure or a Derivative of Image Saliency?](https://openaccess.thecvf.com/content/CVPR2021/html/Song_Mesh_Saliency_An_Independent_Perceptual_Measure_or_a_Derivative_of_CVPR_2021_paper.html) |  |


</details>

<details open id="saliency-video-methods">
<summary>Video</summary>
<br>

| Conference / Journal | Title | Links |
|----------------------|-------|-------|
| TITS 2025 | [MTSF: Multi-Scale Temporal–Spatial Fusion Network for Driver Attention Prediction](https://doi.org/10.1109/TITS.2024.3510116) | [GitHub](https://github.com/JiBingdong/MTSF) |
| TCSVT 2025 | [TM2SP: A Transformer-based Multi-Level Spatiotemporal Feature Pyramid Network for Video Saliency Prediction](https://ieeexplore.ieee.org/abstract/document/10841372) | – |
| TCSVT 2025 | [Toward Unifying Saliency Transformer for Video Saliency Prediction and Detection](https://ieeexplore.ieee.org/abstract/document/10896756) | [Project](https://junwenxiong.github.io/UniST) |
| AAAI 2025 | [CaRDiff: Video Salient Object Ranking Chain of Thought Reasoning for Saliency Prediction with Diffusion](https://ojs.aaai.org/index.php/AAAI/article/view/32785) | – |
| TIV 2024 | [SCOUT+: Towards Practical Task-Driven Drivers’ Gaze Prediction](https://doi.org/10.1109/IV55156.2024.10588743) | [GitHub](https://github.com/ykotseruba/SCOUT) |
| TIV 2024 | [Understanding and Modeling the Effects of Task and Context on Drivers’ Gaze Allocation](https://doi.org/10.1109/IV55156.2024.10588589) | [GitHub](https://github.com/ykotseruba/SCOUT) |
| TMM 2023 | [Multi-Scale Spatiotemporal Feature Fusion Network for Video Saliency Prediction](https://ieeexplore.ieee.org/document/10269025) | – |
| TMM 2023 | [Spatio-Temporal Self-Attention Network for Video Saliency Prediction](https://arxiv.org/pdf/2108.10696v2) | [GitHub](https://github.com/come880412/STSANet) |
| TCSVT 2023 | [Transformer-Based Multi-Scale Feature Integration Network for Video Saliency Prediction](https://ieeexplore.ieee.org/document/10130326) | [GitHub](https://github.com/wusonghe/TMFI-Net) |
| WACV 2023 | [TinyHD: Efficient Video Saliency Prediction with Heterogeneous Decoders using Hierarchical Maps Distillation](https://arxiv.org/pdf/2301.04619v1) | [GitHub](https://github.com/feiyanhu/tinyhd) |
| TITS 2023 | [Driving Visual Saliency Prediction of Dynamic Night Scenes via a Spatio-Temporal Dual-Encoder Network](https://doi.org/10.1109/TITS.2023.3323468) | [GitHub](https://github.com/taodeng/DrFixD-night) |
| TII 2022 | [On-Device Saliency Prediction Based on Pseudoknowledge Distillation](https://ieeexplore.ieee.org/abstract/document/9720080) | [GitHub](https://github.com/chakkritte/PKD) |
| Neurocomputing 2022 | [ECANet: Explicit Cyclic Attention-Based Network for Video Saliency Prediction](https://www.sciencedirect.com/science/article/pii/S0925231221015022) | – |
| TITS 2022 | [Adaptive Short-Temporal Induced Aware Fusion Network for Predicting Attention Regions Like a Driver (ASIAF-Net)](https://doi.org/10.1109/TITS.2022.3165619) | [GitHub](https://github.com/liuchunsense/ASIAFnet) |
| TITS 2022 | [DADA: Driver Attention Prediction in Driving Accident Scenarios](https://doi.org/10.1109/TITS.2020.3044678) | [GitHub](https://github.com/JWFangit/LOTVS-DADA) |
| TCSVT 2022 | [Viewing Behavior Supported Visual Saliency Predictor for 360 Degree Videos](https://ieeexplore.ieee.org/abstract/document/9606880) | [GitHub](https://github.com/vhchuong/Saliency-prediction-for-360-degree-video) |
| TITS 2021 | [HammerDrive: A Task-Aware Driving Visual Attention Model](https://doi.org/10.1109/TITS.2021.3055120) | – |
| PR 2021 | [Video Saliency Prediction Using Enhanced Spatiotemporal Alignment Network](https://arxiv.org/pdf/2001.00292) | [GitHub](https://github.com/cj4L/ESAN-VSP) |
| IJCV 2021 | [Hierarchical Domain-Adapted Feature Learning for Video Saliency Prediction](https://link.springer.com/article/10.1007/s11263-021-01519-y) | [GitHub](https://github.com/perceivelab/hd2s) |
| Scientific Reports 2021 | [Deep Saliency Models Learn Low-, Mid-, and High-Level Features to Predict Scene Attention](https://www.nature.com/articles/s41598-021-97879-z) | – |
| ICPR 2021 | [ATSal: An Attention Based Architecture for Saliency Prediction in 360 Videos](https://link.springer.com/chapter/10.1007/978-3-030-68796-0_22) | [GitHub](https://github.com/mtliba/ATSal) |
| IJCAI 2021 | [GASP: Gated Attention for Saliency Prediction](https://arxiv.org/pdf/2206.04590v1) | [GitHub](https://github.com/knowledgetechnologyuhh/gasp) |
| PR 2020 | [DeepCT: A Novel Deep Complex-Valued Network with Learnable Transform for Video Saliency Prediction](https://www.sciencedirect.com/science/article/pii/S0031320320300406) | – |
| TMM 2020 | [Viewport-Dependent Saliency Prediction in 360° Video](https://ieeexplore.ieee.org/abstract/document/9072511) | – |
| TIP 2020 | [A Spatial-Temporal Recurrent Neural Network for Video Saliency Prediction](https://ieeexplore.ieee.org/document/9263359) | [GitHub](https://github.com/zhangkao/IIP_STRNN_Saliency) |
| AAAI 2020 | [SalSAC: A Video Saliency Prediction Model with Shuffled Attentions and Correlation-Based ConvLSTM](https://ojs.aaai.org/index.php/AAAI/article/view/6927) | – |
| ECCV 2020 | [Unified Image and Video Saliency Modeling](https://arxiv.org/abs/2003.05477) | [GitHub](https://github.com/rdroste/unisal) |
| TIP 2019 | [Video Saliency Prediction Using Spatiotemporal Residual Attentive Networks](https://ieeexplore.ieee.org/document/8811731) | [GitHub](https://github.com/ashleylqx/STRA-Net) |
| BMVC 2019 | [Simple vs Complex Temporal Recurrences for Video Saliency Prediction](https://arxiv.org/pdf/1907.01869v4) | [GitHub](https://github.com/Linardos/SalEMA) |
| BMVC 2019 | [Temporal Recurrences for Video Saliency Prediction](https://doras.dcu.ie/23543/1/_BMVC__Simple_vs_complex_temporal_recurrences_for_video_saliency_prediction__Copy_.pdf) | [GitHub](https://github.com/juanjo3ns/SalBCE) |
| ICCV 2019 | [TASED-Net: Temporally-Aggregating Spatial Encoder-Decoder Network for Video Saliency Detection](https://arxiv.org/abs/1908.05786) | [GitHub](https://github.com/MichiganCOG/TASED-Net) |
| CVPR 2019 | [Understanding and Visualizing Deep Visual Saliency Models](https://openaccess.thecvf.com/content_CVPR_2019/html/He_Understanding_and_Visualizing_Deep_Visual_Saliency_Models_CVPR_2019_paper.html) | [GitHub](https://github.com/SenHe/uavdvsm) |
| TCSVT 2018 | [Learning Coupled Convolutional Networks Fusion for Video Saliency Prediction](https://ieeexplore.ieee.org/abstract/document/8474977) | – |
| TCSVT 2018 | [Video Saliency Prediction Based on Spatial-Temporal Two-Stream Network](https://ieeexplore.ieee.org/document/8543830) | [GitHub](https://github.com/zhangkao/IIP_TwoS_Saliency) |
| TIP 2018 | [Find Who to Look at: Turning From Action to Saliency](https://ieeexplore.ieee.org/abstract/document/8360155) | [GitHub](https://github.com/yufanLiu/find) |
| TPAMI 2018 | [Revisiting Video Saliency Prediction in the Deep Learning Era](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8744328) | – |
| TPAMI 2018 | [Predicting the Driver's Focus of Attention: The DR(eye)VE Project](https://doi.org/10.1109/TPAMI.2018.2845370) | [GitHub](https://github.com/ndrplz/dreyeve) |
| ICMEW 2018 | [SalGAN360: Visual Saliency Prediction on 360 Degree Images With Generative Adversarial Networks](https://ieeexplore.ieee.org/abstract/document/8551543) | [GitHub](https://github.com/FannyChao/SalGAN360) |
| ECCV 2018 | [DeepVS: A Deep Learning Based Video Saliency Prediction Approach](http://openaccess.thecvf.com/content_ECCV_2018/papers/Lai_Jiang_DeepVS_A_Deep_ECCV_2018_paper.pdf) | [GitHub](https://github.com/remega/OMCNN_2CLSTM) |
| CVPR 2018 | [Revisiting Video Saliency: A Large-Scale Benchmark and a New Model](https://arxiv.org/abs/1801.07424) | [GitHub](https://github.com/wenguanwang/DHF1K) / [Code](https://github.com/Nablax/ACLnet-Pytorch) |
| CVPR 2018 | [Going From Image to Video Saliency: Augmenting Image Salience With Dynamic Attentional Push](https://openaccess.thecvf.com/content_cvpr_2018/html/Gorji_Going_From_Image_CVPR_2018_paper.html) | – |
| CVPR 2018 | [Cube Padding for Weakly-Supervised Saliency Prediction in 360° Videos](https://arxiv.org/abs/1806.01320v1) | [GitHub](https://github.com/hsientzucheng/CP-360-Weakly-Supervised-Saliency) |
| TIV 2017 | [Learning Where to Attend Like a Human Driver](https://doi.org/10.1109/IVS.2017.7995833) | [GitHub](https://github.com/francescosolera/dreyeving) |
| TMM 2017 | [Spatio-Temporal Saliency Networks for Dynamic Saliency Prediction](https://arxiv.org/abs/1607.04730) | [GitHub](https://github.com/cagdasbak/dynamicsaliency) |
| arXiv 2016 | [Deep Learning for Saliency Prediction in Natural Video](https://arxiv.org/abs/1604.08010) | – |
| ICIP 2016 | [Transfer Learning With Deep Networks for Saliency Prediction in Natural Video](https://ieeexplore.ieee.org/abstract/document/7532629) | – |
| TIP 2014 | [Saliency Prediction on Stereoscopic Videos](https://ieeexplore.ieee.org/abstract/document/6728719) | – |


</details>

<details open id="saliency-other-methods">
<summary>Other</summary>
<br>

| Conference / Journal | Title | Links |
|----------------------|-------|-------|
| CVPR 2024 | [DiffSal: Joint Audio and Video Learning for Diffusion Saliency Prediction](https://arxiv.org/abs/2403.01226) | [GitHub](https://github.com/junwenxiong/diff_sal) |
| CVPR 2023 | [CASP-Net: Rethinking Video Saliency Prediction from an Audio-Visual Consistency Perceptual Perspective](https://arxiv.org/abs/2303.06357) | – |
| IROS 2021 | [ViNet: Pushing the Limits of Visual Modality for Audio-Visual Saliency Prediction](https://ieeexplore.ieee.org/abstract/document/9635989) | [GitHub](https://github.com/samyak0210/ViNet) |
| TIP 2020 | [A Multimodal Saliency Model for Videos With High Audio-Visual Correspondence](https://ieeexplore.ieee.org/document/8962278) | – |
| ECCV 2020 | [Learning to Predict Salient Faces: A Novel Visual-Audio Saliency Model](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650409.pdf) | [GitHub](https://github.com/MinglangQiao/visual_audio_saliency) |
| CVPR 2020 | [STAViS: Spatio-Temporal AudioVisual Saliency Network](https://arxiv.org/pdf/2001.03063) | [GitHub](https://github.com/atsiami/STAViS) |
| arXiv 2019 | [DAVE: A Deep Audio-Visual Embedding for Dynamic Saliency Prediction](https://arxiv.org/abs/1905.10693) | [GitHub](https://github.com/hrtavakoli/DAVE) |
| CVPR 2018 | [Audio-Visual Temporal Saliency Modeling Validated by fMRI Data](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w39/html/Koutras_Audio-Visual_Temporal_Saliency_CVPR_2018_paper.html) | – |
| CVPR 2017 | [Saliency Revisited: Analysis of Mouse Movements Versus Fixations](https://openaccess.thecvf.com/content_cvpr_2017/html/Tavakoli_Saliency_Revisited_Analysis_CVPR_2017_paper.html) | – |
| ECCV 2016 | [Where Should Saliency Models Look Next?](https://link.springer.com/chapter/10.1007/978-3-319-46454-1_49) | – |
| CVPR 2016 | [Spatially Binned ROC: A Comprehensive Saliency Metric](https://openaccess.thecvf.com/content_cvpr_2016/html/Wloka_Spatially_Binned_ROC_CVPR_2016_paper.html) | – |

</details>


</details>

---

---
<a id="scanpath-prediction"></a>
## 🌊 Scanpath Prediction


### 1. Datasets 

<details id="scanpath-prediction-datasets">
<summary>Dataset list</summary>
<br>

_To be added._

</details>

---
<a id="scanpath-prediction-methods"></a>
### 2. Methods 

<details id="scanpath-image-methods">
<summary>Image</summary>
<br>

_To be added._

</details>

<details id="scanpath-3d-methods">
<summary>3D</summary>
<br>

_To be added._

</details>


---
<a id="applications-of-human-visual-attention"></a>
## 🎯 Applications of Human Attention


### 1. Datasets 

<details id="applications-datasets">
<summary>Dataset list</summary>
<br>

_To be added._

</details>

---
<a id="applications-methods"></a>
### 2. Methods 

<details id="applications-hci">
<summary>HCI / Interaction / AR</summary>
<br>

_To be added._

</details>

<details id="applications-ad">
<summary>AD / Cognitive & Neuro</summary>
<br>

_To be added._

</details>

<details id="applications-robotics">
<summary>Robotics</summary>
<br>

_To be added._

</details>

<details id="applications-medicine">
<summary>Medicine</summary>
<br>

_To be added._

</details>

<details id="applications-design">
<summary>Design</summary>
<br>

_To be added._

</details>

<details id="applications-commerce">
<summary>Commerce</summary>
<br>

_To be added._

</details>


---





