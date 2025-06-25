# Comparison of local feature matching algorithms

by
Romy Gros

This repository contains the code used to generate the results presented in the associated manuscript, in which the performance of various local feature matching algorithms have been tested on image pairs including polarimetric images and histological images.

The repository also allows researchers to test the algorithms using their own database.


## Software installation

The [Jupyter notebook](http://jupyter.org/) `compare_methods.ipynb` contains an example run of the comparison of the different methods.
The data used in this study can be downloaded from [this repository](https://osf.io/rsxgp/?view\_only=c1cc83138a494606820e86c173d8d6f2).

It is recommended to use a Python virtual environment to manage dependencies and avoid conflicts with other projects.

1. **Create a virtual environment (replace `venv` with your preferred name):**

    ```sh
    python3 -m venv venv
    ```

2. **Activate the virtual environment:**

    - On Linux/macOS:
        ```sh
        source venv/bin/activate
        ```
    - On Windows:
        ```sh
        venv\Scripts\activate
        ```

3. **Download the repository and install the required dependencies:**

    ```sh
    git clone https://github.com/eleagros/clfm.git
    cd clfm
    pip install -e .
    ```

4. **Install PyTorch:**

    Follow the instructions for your system and CUDA version at [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).  
    For example, for Linux with pip and CUDA 11.8:

    ```sh
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

## Third-party dependencies and licensing

Some third-party code is provided in `third_party/` is under different licenses (MIT, Apache 2.0, or academic/non-commercial).

**Important notice: SuperGlue** (by Magic Leap, Inc.) is under a restrictive academic/non-commercial license.  

### How to obtain and use SuperGlue

1. **Download the original SuperGlue code**  
   Get a clean, unmodified copy from the official repository and place it in the `third_party/SuperGluePretrainedNetwork` folder:  
   [https://github.com/magicleap/SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork)

2. **Apply the patch provided**  
   If this repository provides a patch file (e.g., `superglue.patch`), save it in the root directory of the original SuperGlue code and run:

   ```sh
   patch third_party/SuperGluePretrainedNetwork/demo_superglue.py < third_party/superglue.patch
   ```

**Note:**  
- Only the patch file and instructions are provided here, not the SuperGlue code itself.
- This approach respects the license and allows reproducibility.

### Additional requirement: Fiji

To use the image alignment and processing features, you need to download [Fiji](https://imagej.net/software/fiji/downloads) (a distribution of ImageJ).  
After downloading, **extract the `Fiji.app` folder and place it inside the `third_party/` directory** of this repository:

```
third_party/Fiji.app/
```

This ensures all Fiji/ImageJ-based scripts will work correctly.

## Data

The dataset contains two sub-folders: `histology` and `polarimetry`.

- `histology`: Contains histological images of the samples.
- `polarimetry`: Contains polarimetric images of the samples.

To reproduce the results or test the algorithms on your own data, organize your data in the same structure.

The code to produce the ground truth manually (if not provided via resampling maps) is provided in the `manual_step.py` file.
An example is provided in the notebook `generate_ground_truth.ipynb`.

## License
All source code is made available under a BSD license. See `LICENSE` for the full license text.

## Citation
If you use this code or data, please cite the associated manuscript.

> **Note:** There is a known issue with Java compatibility on some machines when using Fiji/ImageJ and PyImageJ.  
> To avoid these issues, it is recommended to install PyImageJ and set up your environment using the instructions at [https://py.imagej.net/en/latest/Install.html#installing-via-conda-mamba](https://py.imagej.net/en/latest/Install.html#installing-via-conda-mamba).  
> Please use the provided conda environment for installing this repository if you encounter Java-related problems.