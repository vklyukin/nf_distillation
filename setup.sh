mkdir -p ./data/weights

wget https://www.dropbox.com/s/3hkko8rcnof7edg/notop_teacher_0.pth?dl=0 -O ./data/weights/notop_teacher_0.pth
wget https://download.pytorch.org/models/vgg16-397923af.pth -O ./data/weights/vgg16-397923af.pth
wget https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth -O ./data/weights/inception_v3.pth
