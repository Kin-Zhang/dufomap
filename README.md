DUFOMap Python Package
---

[![arXiv](https://img.shields.io/badge/arXiv-2403.01449-b31b1b.svg)](https://arxiv.org/abs/2403.01449) 
[![page](https://img.shields.io/badge/Web-Page-green)](https://kin-zhang.github.io/dufomap)
[![Stable Version](https://img.shields.io/pypi/v/dufomap?label=stable)](https://pypi.org/project/dufomap/#history)
[![Python Versions](https://img.shields.io/pypi/pyversions/dufomap)](https://pypi.org/project/dufomap/)
[![Download Stats](https://img.shields.io/pypi/dm/dufomap)](https://pypistats.org/packages/dufomap)

Author: [Qingwen Zhang](https://kin-zhang.github.io/). Please give us a star if you like this repo! 🌟 and [cite our work](#acknowledgement) 📖 if you find this useful for your research. Thanks!

Available in: <a href="https://github.com/Kin-Zhang/dufomap"><img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" /></a> <a href="https://github.com/Kin-Zhang/dufomap"><img src="https://img.shields.io/badge/Windows-0078D6?st&logo=windows&logoColor=white" /> 

All datasets and benchmark methods check: [DynamicMap_Benchmark](https://github.com/KTH-RPL/DynamicMap_Benchmark) if you are not only interested in the method but also the comparison.

Note: to know better about the parameters meaning and ablation study, please check the [paper](https://arxiv.org/abs/2403.01449).

📜 Change Log:
- 2024-08-29: Remove OpenMP but add oneTBB for all codes. Speed up the code by 18%. Check the [discussion here](https://github.com/Kin-Zhang/dufomap/discussions/1).
- 2024-08-28: Refactor the code and add `__init__.py` to have the input array must be contiguous first.
- 2024-07-03: Speed up nanobind `np.array` <-> `std::vector<Eigen:: Vector3d>` conversion and also `NOMINSIZE` in make. Speed difference: 0.1s -> 0.01s. Based on [discussion here](https://github.com/wjakob/nanobind/discussions/426).
- 2023-11-28: Initial version.

Installation:

```bash
pip install dufomap
```

## Run the example

Demo usage:
```python
from dufomap import dufomap
# pointcloud: Nx3 numpy array
# pose: 4x4 numpy array or a list with 7 elements (x,y,z,qw,qx,qy,qz)
mydufo = dufomap()
mydufo.run(pointcloud, pose, cloud_transform=True)
label = mydufo.segment(pointcloud, pose, cloud_transform = True)
# 1: dynamic, 0: static
```

Or you can check the full example script in [example.py]. If you run the example script, it will directly show a default effect of demo data.

```bash
# for this demo you need install open3d to run the visualization
pip install open3d fire

wget https://zenodo.org/records/10886629/files/00.zip
unzip 00.zip
python example.py --data_dir ./00
```

![dufomap_py](https://github.com/user-attachments/assets/bc921c8d-0dbd-4813-9c09-9ad0d051e71d)


## Acknowledgement

This python binding is developed during our SeFlow work, please cite our paper if you use this package in Python:

```bibtex
@article{zhang2024seflow,
  author={Zhang, Qingwen and Yang, Yi and Li, Peizheng and Andersson, Olov and Jensfelt, Patric},
  title={SeFlow: A Self-Supervised Scene Flow Method in Autonomous Driving},
  journal={arXiv preprint arXiv:2407.01702},
  year={2024}
}
@article{daniel2024dufomap,
  author={Duberg, Daniel and Zhang, Qingwen and Jia, Mingkai and Jensfelt, Patric},
  journal={IEEE Robotics and Automation Letters}, 
  title={{DUFOMap}: Efficient Dynamic Awareness Mapping}, 
  year={2024},
  volume={9},
  number={6},
  pages={5038-5045},
  doi={10.1109/LRA.2024.3387658}
}
```