# LEMON

Official code for the 3DV 2025 submission LEMON: Localized Editing with Mesh Optimization and Neural Shaders.

## Installation

Setup the environment and install basic requirements using conda

```bash
conda env create -f environment.yml
conda activate lemon
```

### Nvdiffrast

To install [Nvdiffrast](https://github.com/NVlabs/nvdiffrast) from source, run the following in the main directory:

```bash
git clone https://github.com/NVlabs/nvdiffrast.git
cd nvdiffrast
python -m pip install .
```

### pyremesh

Option 1 (preferred): Install [pyremesh](https://github.com/sgsellan/botsch-kobbelt-remesher-libigl) from pre-built packages in the `pyremesh` subdirectory.

From the main directory, run:

```bash
python -m pip install --no-index --find-links ./ext/pyremesh pyremesh
```

Option 2: Install pyremesh from source.

Follow the instructions at https://github.com/sgsellan/botsch-kobbelt-remesher-libigl.

### nerfstudio[gen]

If you get "Missing Stable Diffusion packages." error, you can resolve it by installing `nerfstudio[gen]`.

```bash
pip install nerfstudio[gen]
```

## Reconstructing DTU Scans

We provide 2 DTU scans and corresponding vertex scores for inputs. You can also download the [full dataset](https://www.dropbox.com/s/56ym2qrjfg7jymo/data.zip) (2.3 GB) (Taken from Neural Deferred Shading github) and resize images to 512x384(set camera parameters accordingly to that) to get our results. First pre-processing of mesh for vertex scores could take time.

We provide three examples. To start the run on the one of the examples:

```bash
python mcnet-reconstruct.py --config ./configs/global/skull-clown-config.yaml
python mcnet-reconstruct.py --config ./configs/local/owl-monkey-config.yaml
python mcnet-reconstruct.py --config ./configs/local/stand-drawer-config.yaml
```

## Using the Interactive Viewer of NDS

You can also look at the reconstructed mesh and neural shader using interactive viewer provided by authors of NDS Before you can launch the viewer, install the additional dependencies with

```bash
conda activate lemon
pip install glfw==2.5.3 moderngl==5.6.4 pyrr==0.10.3 pyopengl==3.1.6
```

The `pycuda` dependency needs to be build from source with OpenGL support. In your preferred directory, run

```bash
git clone --recursive https://github.com/inducer/pycuda.git
cd pycuda
git checkout v2022.1

conda activate instruct-nds
python ./configure.py --cuda-enable-gl
python setup.py install
```

The viewer is launched by running the python script `view.py`, providing the mesh, the neural shader and a bounding box as input. For example, the reconstruction results for the DTU skull can be viewed by running

```bash
python view.py --mesh out/65_skull/clown/meshes/mesh_002000.obj --shader out/65_skull/clown/shaders/shader_002000.pt --bbox out/65_skull/clown/bbox.txt
```

## Citation

All of the NDS dependencies are taken from the original paper.

```bibtex
@InProceedings{worchel:2022:nds,
      author    = {Worchel, Markus and Diaz, Rodrigo and Hu, Weiwen and Schreer, Oliver and Feldmann, Ingo and Eisert, Peter},
      title     = {Multi-View Mesh Reconstruction with Neural Deferred Shading},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      month     = {June},
      year      = {2022},
      pages     = {6187-6197}
}
```

## Troubleshooting of NDS

### CUDA Out of Memory

The reconstruction and editing can be quite heavy on GPU memory. We have used 512x384 images for our experiments. (Other methods mostly use 512x512 but we wanted to keep aspect ratio of DTU images)

### IMPORTANT: Initial Reconstruction Hangs at Remeshing or High Loss After Remeshing

While the remeshing step can take some time especially at higher mesh resolutions, it sometimes hangs indefinitely. This issue comes from calling the function `remesh_botsch` in the `pyremesh` package, which does not return.

For now, the reconstruction has to be aborted and restarted.

Sometimes the loss gets really high and mesh becomes bad after a remeshing . Check the loss after last remeshing step(e.g. 1500) if it is too high. We suggest restarting or increasing the normal loss hyperparameter.
