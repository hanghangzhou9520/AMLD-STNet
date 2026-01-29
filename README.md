# [AMLD-STNet: Adaptive Multi-scale Lagrange Dynamics Spatial-Temporal Network for 3D Skeleton-based Human Motion Prediction](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11363270) (TCSVT 2026)

>[Hanghang Zhou](https://orcid.org/0000-0002-5046-1935), [Yumei Zhang](https://orcid.org/0000-0002-2595-3276)<sup>*</sup>,
> [Xiangying Guo](https://orcid.org/0000-0002-9887-2338), [Keying Zhao](https://orcid.org/0009-0006-8710-7696),  [Honghong Yang](https://orcid.org/0000-0002-4124-5317), [Xiaojun Wu](https://orcid.org/0000-0002-7779-553X),  [Zexing Du](https://orcid.org/0000-0002-7779-553X).
## Abstract
> Human body dynamics, as a temporal variation pattern of pose sequences in 3D skeleton-based human motion prediction, 
> has been extensively studied in spatial-temporal dependent modeling of deep learning. However, designing an effective
> modeling approach that fully harnesses physical principles to enhance algorithmic performance remains a challenge. 
> Existing approaches prioritize displacement information, processing deterministic physical parameters via standard neural networks 
> while modeling rotation motion through simplified angular constraints. Such physical approximation methods neglect the 
> high-dimensional and dynamic characteristics of Dynamics variables, undermining the integrity and diversity of human motion
> feature representations. To alleviate these limitations, we propose an Adaptive Multi-scale Lagrange Dynamics Spatial-Temporal 
> Network (AMLD-STNet), which directly embeds learnable neural network modules within physical equations to activate multi-scale 
> dynamic physical feature modeling of human motion. Specifically, A Lagrange Dynamics Network (LD-Net) is constructed, which 
> designs a set of joint force adjacency matrices to analyze the mechanical correlation between the velocity and acceleration 
> of each joint motion through the Lagrange Dynamics equation. Subsequently, the Lagrange Dynamic Spatial-Temporal Network (LD-STNet) 
> is established, which utilizes LD-Net to extract multi-perspective high-dimensional features of human displacement and rotational 
> motion represented by Dynamics pose variables. To capture the mechanical correlation of joint node groups, we design a multi-scale 
> streams LD-STNet, which can realize adaptive scale transformation according to the joint force adjacency. Additionally, 
> Euler angle loss is employed to enforce rotational consistency constraints, thereby enhancing physical realism during network
> training. Finally, extensive experiments are conducted on three popular benchmarks, such as Human 3.6M, AMASS, and 3DPW, 
> among which AMLD-STNet achieved state-of-the-art results with a smaller model size.
