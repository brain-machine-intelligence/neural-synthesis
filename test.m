function test_example_CNN

addpath('../NN','../util','../CNN', '../data')
load mnist_uint8;
ii=1;
train_x = double(reshape(train_x',28,28,60000))/255;
test_x = double(reshape(test_x',28,28,10000))/255;
train_y = double(train_y');
test_y = double(test_y');

%% ex1 Train a 6c-2s-12c-2s Convolutional neural network 
%will run 1 epoch in about 200 second and get around 11% error. 
%With 100 epochs you'll get around 1.2% error
rand('state',0)
cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};

cnn = cnnsetup(cnn, train_x, train_y);

opts.alpha = 1;
opts.batchsize = 50;
opts.numepochs = 2;

cnn = cnntrain(cnn, train_x, train_y, opts);

[er1, bad] = cnntest(cnn, test_x, test_y);

%plot mean squared error

    for pp = 2 : numel(cnn.layers)
        if strcmp(cnn.layers{pp}.type, 'c')
            CNN_W{pp,ii} = cnn.layers{pp}.k;
            CNN_B{pp,ii} = cnn.layers{pp}.b;
        end
    end

    
    CNN_FW(:,:,ii) = cnn.ffW;
    CNN_FFB(:,:,ii) = cnn.ffb;
    
    new_cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};
new_cnn = cnnsetup(new_cnn, train_x, train_y);

for pp = 2 : numel(new_cnn.layers)
       if strcmp(cnn.layers{pp}.type, 'c')
           new_cnn.layers{pp}.k =  CNN_W{pp,ii};
           new_cnn.layers{pp}.b =  CNN_B{pp,ii};
       end
 end

    
    new_cnn.ffW = CNN_FW(:,:,ii);
    new_cnn.ffb = CNN_FFB(:,:,ii);

[er2, bad] = cnntest(new_cnn, test_x, test_y);


er1
er2

figure; plot(cnn.rL);

assert(er1<0.12, 'Too big error');