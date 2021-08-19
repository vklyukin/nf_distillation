mkdir -p ./data/weights
mkdir -p ./data/data

wget https://www.dropbox.com/s/3hkko8rcnof7edg/notop_teacher_0.pth?dl=0 -O ./data/weights/notop_teacher_0.pth
wget https://download.pytorch.org/models/vgg16-397923af.pth -O ./data/weights/vgg16-397923af.pth
wget https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth -O ./data/weights/inception_v3.pth
wget -O processed_data.tar.gz https://zenodo.org/record/1161203/files/data.tar.gz?download=1
tar -xvf processed_data.tar.gz -C ./data/data/ --strip-components=1
rm processed_data.tar.gz
rm -rf ./data/data/cifar10
rm -rf ./data/data/mnist
