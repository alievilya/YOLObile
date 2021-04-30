


sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev
# install gdown to download from Google drive, if not done yet
sudo -H pip3 install gdown
# copy binairy
sudo cp ~/.local/bin/gdown /usr/local/bin/gdown
# download TorchVision 0.8.2
gdown https://drive.google.com/uc?id=1Z14mNdwgnElOb_NYkRaDCwP31scd7Mfz
# install TorchVision 0.8.2
sudo -H pip3 install torchvision-0.8.2a0+2f40a48-cp36-cp36m-linux_aarch64.whl
# clean up
rm torchvision-0.8.2a0+2f40a48-cp36-cp36m-linux_aarch64.whl
