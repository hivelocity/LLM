## Installing nvidia container runtime
These steps are adapted for Ubuntu 22.04

1. Remove existing drivers if current ones are old or unapplicable:
    - Purge all drivers

          sudo dpkg --purge *nvidia*
    - Check `dpkg` if anything nvidia is leftover. I had one remaining one left during test

          dpkg -l | grep nvidia
        - You may get output that looks something like this (don't expect yours to match):

              $ dpkg -l | grep -i nvidia
              ii  libnvidia-example1:amd64                                  1.14.3-1                                                       amd64        NVIDIA container runtime library
              rc  nvidia-example2                                           1.14.3-1                                                       amd64        NVIDIA Container Toolkit Base
            - Packages mentioned here should only be removed if they mention `nvidia` in its entirety, packages like
              `nouveau` should be left alone.
            - Here I would run `sudo apt remove --purge libnvidia-example1:amd64` and repeat for `nvidia-example2`. Copy
              paste any packages that appear here

    - Autoremove to remove anything else

          sudo apt autoremove
    - Reboot and in the shell type `nv` and hit tab to make sure there no nvidia executables. At time of testing only
      one executable came up, and that was `nvidia-detect` which is part of ubuntu so this was fine.

2. Install NVIDIA drivers
  - [Download drivers](https://www.nvidia.com/en-us/drivers/unix/) under "Linux x86_64/AMD64/EM64T", `chmod +x` to make executable and run.
    - When asked about disabling the default nouvaeu drivers, select yes
    - It *may* fail the installation intentionally after disabling nouveau, if so simply reboot
    - Questions about 32-bit compatibility, enable Xorg driver, etc should be Yes
    - Version installed when readme was written was 535.146.02
  - Reboot and run `nvidia-smi`, you should get output like this:

        $ nvidia-smi
        Fri Dec  8 10:07:57 2023       
        +---------------------------------------------------------------------------------------+
        | NVIDIA-SMI 535.146.02             Driver Version: 535.146.02   CUDA Version: 12.2     |
        |-----------------------------------------+----------------------+----------------------+
        | GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
        | Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
        |                                         |                      |               MIG M. |
        |=========================================+======================+======================|
        |   0  NVIDIA GeForce RTX 3090        Off | 00000000:08:00.0 Off |                  N/A |
        |  0%   53C    P8               8W / 350W |     76MiB / 24576MiB |      0%      Default |
        |                                         |                      |                  N/A |
        +-----------------------------------------+----------------------+----------------------+
                                                                                                 
        +---------------------------------------------------------------------------------------+
        | Processes:                                                                            |
        |  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
        |        ID   ID                                                             Usage      |
        |=======================================================================================|
        |    0   N/A  N/A      1478      G   /usr/lib/xorg/Xorg                           56MiB |
        |    0   N/A  N/A      1599      G   /usr/bin/gnome-shell                         12MiB |
        +---------------------------------------------------------------------------------------+


3. Install docker
  - Recommend installing from docker installation instructions, below is a copy of what was followed from https://docs.docker.com/engine/install/ubuntu/
during testing
    - Remove conflicting packages:

          for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done```
    - Install setup docker APT repo:

          # Add Docker's official GPG key:
          sudo apt-get update
          sudo apt-get install ca-certificates curl gnupg
          sudo install -m 0755 -d /etc/apt/keyrings
          curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
          sudo chmod a+r /etc/apt/keyrings/docker.gpg

          # Add the repository to Apt sources:
          echo \
            "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
            $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
            sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
          sudo apt-get update
    - Install latest version:
    
          sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    - Test docker with

          docker run --rm ubuntu:latest echo hello
    - Versions installed when readme was written:

          $ dpkg -l | grep -i docker
          ii  docker-buildx-plugin                                        0.11.2-1~ubuntu.22.04~jammy                                    amd64        Docker Buildx cli plugin.
          ii  docker-ce                                                   5:24.0.7-1~ubuntu.22.04~jammy                                  amd64        Docker: the open-source application container engine
          ii  docker-ce-cli                                               5:24.0.7-1~ubuntu.22.04~jammy                                  amd64        Docker CLI: the open-source application container engine
          ii  docker-ce-rootless-extras                                   5:24.0.7-1~ubuntu.22.04~jammy                                  amd64        Rootless support for Docker.
          ii  docker-compose-plugin                                       2.21.0-1~ubuntu.22.04~jammy                                    amd64        Docker Compose (V2) plugin for the Docker CLI.

4. Install nvidia container driver
   - Below are steps copied from https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt
   - Install APT repository:

            curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
              && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
                sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   - Update package list

            sudo apt-get update
   - Install the NVIDIA Container Toolkit packages

            sudo apt-get install -y nvidia-container-toolkit
   - Configure applicable runtimes (docker and containerd were the only ones applicable at time of writing readme):
     - Docker:
       - Configure:

              sudo nvidia-ctk runtime configure --runtime=docker
       - Restart docker:

              sudo systemctl restart containerd
     - containerd:
       - Configure

                sudo nvidia-ctk runtime configure --runtime=crio
       - Restart containerd

                sudo systemctl restart containerd

    - Verify nvidia contain driver works:

            sudo docker run --rm --gpus all ubuntu:latest nvidia-smi

      - Should get similiar output to `nvidia-smi` command outside of docker
      - If you get this error:

            docker: Error response from daemon: failed to create task for container: failed to create shim task: OCI runtime create failed: runc create failed: unable to start container process: exec: "nvidia-smi": executable file not found in $PATH: unknown.

        - This was because I didn't upgrade my docker version. Follow the docker steps to get the latest docker
        

## Running Example

The `docker-compose.yml` should already specify GPU device access so build with:

    docker compose build

Newer versions of docker have deprecated `docker-compose` in favor of the `docker compose` subcommand, so make sure
you use that verison in case you get an error like this:

    ERROR: The Compose file './docker-compose.yml' is invalid because:
    services.torch.deploy.resources.reservations value Additional properties are not allowed ('devices' was unexpected)

Put your model files in the model directory like [Mistral Orca](https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca)

    cd models/
    git lfs install  # enable large file support 
    git clone https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca

May take a while (~14 GB). Latest commit of model repository was 4a37328cef00f524d3791b1c0cc559a3cc6af14d

`cd ..`  back up to main directroy and test model inference with prompt script:

    docker-compose run --rm torch /prompt.py -l 2048 --half

`--half` is needed for this specific model since it's larger than available VRAM on a 4090 (24GB) `-l 2048` sets
max generated token length

Example output:

    docker-compose run --rm torch /prompt.py -l 2048 --half
    Creating llm-test_torch_run ... done
    Namespace(model='/models/Mistral-7B-OpenOrca', half=True, prompt='A chat.', max_length=2048)
    BasicConfig(max_length=2048, temperature=1.1, top_p=0.95, repetition_penalty=1.0)
    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
    Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:06<00:00,  3.29s/it]
    > write a recipe for disaster

    Title: The Recipe for Disaster

    Ingredients:
    - 1 bottle expired mayonnaise
    - 1 can spoiled tuna fish
    - 1 loaf moldy bread
    - 1 handful of expired spaghetti noodles
    - 1 bottle of rotten ketchup

    (...omitted)


Note: If there is only one model in the `models` directory, you don't have to specify the path, but if using multiple models,
you may need the `--model` or `-m` switch to specify your model path like:

    docker-compose run --rm torch -m /models/llama.cpp /prompt.py -l 2048 --half