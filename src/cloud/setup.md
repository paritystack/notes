# Setup

## Setup GPU instances

Make sure the hardisk size is at least 30GB

```bash
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py 

#if required Change driver version in the py file from (DRIVER_VERSION = "525.125.06") to	550.54.15
sed -i 's/525.125.06/550.54.15/' install_gpu_driver.py

#run the script
sudo apt install python3-venv python3-dev
sudo python3 install_gpu_driver.py

#verify the installation
nvidia-smi

#install pytorch
pip3 install torch torchvision torchaudio

#install cuda toolkit
sudo apt install nvidia-cuda-toolkit
nvcc --version
```


Swap file
```bash
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## Where this connects

- [AWS](aws.md) — AWS GPU instance setup is covered in setup.md
- [Google Cloud](google_cloud.md) — GCP GPU instances as an alternative
- [CUDA](../machine_learning/cuda.md) — the GPU programming these instances are provisioned for
- [Docker](../tools/docker.md) — containerizing workloads on cloud instances
