# GANSynth
The code in this repository is a modified version of GANSynth

[Github Repository](https://github.com/magenta/magenta/tree/f73ff0c91f0159a925fb6547612199bb7c915248/magenta/models/gansynth)

[Paper](https://arxiv.org/pdf/1902.08710.pdf)

# Uncomprehensive list of modifications
- not running the 2nd iteration over blocks to_rgb (https://github.com/magenta/magenta/blob/f73ff0c91f0159a925fb6547612199bb7c915248/magenta/models/gansynth/lib/networks.py#L397)
- not using alpha blending