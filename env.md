# Environment

Follow these steps to create a fully functional environment for this project.

- Create a Python virtual environment
- Install requirements in <code>requirements.txt</code>
  Note: if <code>SimpleITK</code> gets stuck in the wheel creation process, install it using <code>conda</code> rather than <code>pip</code>
- Open the following file in your preferred text editor: <code><YOUR_PYTHON_ENV_DIR>/lib/pythonX.YZ/site-packages/medpy/metric</code>
- Replace all occurrences of <code>np.bool</code> with <code>bool</code>
- Save

The last steps are needed to have a properly working <code>inference</code> code and to keep good performances on modern (series 30XX and up) NVIDIA GPUs.
The [repo](https://github.com/Beckschen/TransUNet) from which we took part of the TransU-Net original code uses very old PyTorch and TorchVision versions which are not properly comaptible with such modern devices. 
