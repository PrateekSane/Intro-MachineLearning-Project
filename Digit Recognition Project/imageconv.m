I = imread('6greyscale20x20.jpg');
[imgdata, c] = gray2ind(I, 64);
imgdata=imgdata(:,:,1);
imgdata=reshape(imgdata, 1, []);

%save('imgdata.mat', 'imgdata');
