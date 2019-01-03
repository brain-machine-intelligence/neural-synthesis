function Paper_figure_CNN_ImageNet_basic_training()
addpath('./NN','./util')
rep = 1;
%DL

 %convnet = vgg19;

avr_num = 30; % 반복성능을 비교하기 위해서 

train_size = 32;
init_lr = 0.7;
init_nlr = 0.2;
learning_rate = init_lr;
batch_norm = 0;

opts.numepochs =  30;                %  Number of full sweeps through data

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % 어떤 클래스를 가져온것인가
% 
%  trainingFeatures = trainingFeatures';
%  testFeatures = testFeatures';
% % 
%  mean_train_x = mean(trainingFeatures,1);
%  std_train_x = std(trainingFeatures);
% % 
%  trainingFeatures = (trainingFeatures - repmat(mean_train_x,[size(trainingFeatures,1),1]))./repmat(std_train_x, [size(trainingFeatures,1), 1]);
%  testFeatures = (testFeatures - repmat(mean_train_x,[size(testFeatures,1),1]))./repmat(std_train_x, [size(testFeatures,1),1]);

load('data_0806_vgg19.mat');

trainN_x{1} = trainingFeatures(1:train_size,:);
trainN_x{2} =  trainingFeatures(size(trainingFeatures,1)/2+1:size(trainingFeatures,1)/2+train_size,:);

testN_x = {};

testN_x{1} = testFeatures(1:size(testFeatures,1)/2,:);
testN_x{2} = testFeatures(size(testFeatures,1)/2+1:end,:);

% 2개의 클래스의 정답.
Train_N{1} = [ones(60000,1) zeros(60000,1)];
Train_N{2} = [zeros(60000,1) ones(60000,1)];

train_num = train_size;
    
    trainSet_x = [];
    trainSetN_x = {};
       
    trainSet_y = [];
    
    testSet_x = [];
    testSet_y = [];
    
    for class_num = 1:2
        m = size(trainN_x{class_num}, 1);
        kk1 = randperm(m);
        trainSet_x = [trainSet_x; trainN_x{class_num}(kk1(1:train_num),:);];
        trainN_x{class_num} = trainN_x{class_num}(kk1(1:train_num),:);
        trainSet_y = [trainSet_y; Train_N{class_num}(1:train_num,:);];
        
        trainSetN_x{class_num} = trainN_x{class_num}(kk1(1:train_num),:);
        testSet_x = [testSet_x; testN_x{class_num}(:,:);];
        testSet_y = [testSet_y; Train_N{class_num}(1:size(testN_x{class_num},1),:);];
    end

    
    %batch normalization을 위한 0~1로 만들기
    train_x = double(trainSet_x);
    test_x  = double(testSet_x);

    %뒤의 재학습에 사용할 데이터
    pre_x = train_x;
    preTest_x = test_x;
    train_y = trainSet_y;
    test_y = testSet_y;
    

  % makeImage(train_x);
% normalize

[train_x, mu, tanh_opta] = zscore(pre_x);
test_x = normalize(preTest_x, mu, tanh_opta);

regenerateNN_err = cell(1,avr_num);
basicNN_err = cell(1,avr_num);


%% ex1 vanilla neural net
%rand('state',0)
err = zeros(1,avr_num);
nn_err = zeros(avr_num, opts.numepochs);
beforeErr = [];
for ii = 1:avr_num
nn = nnsetup([size(trainingFeatures,2) 250 50 2]);
nn.activation_function = 'tanh_opt';    %  tanh_optoid activation function
nn.learningRate = learning_rate;                %  tanh_opt require a lower learning rate
nn.norm_learningRate = init_nlr;
nn.batch_normalize = batch_norm;
opts.batchsize = 4;               %  Take a mean gradient step over this many samples 
opts.validation = 1;
%[nn, ~] = nntrain(nn, train_x, train_y, opts);

% data shuffling

[nn, ~] = nntrain(nn, train_x, train_y, opts, test_x, test_y);
[err_tmp, ~] = nntest(nn,test_x,test_y);
err(1,ii) = err_tmp;
nn_err(ii,:) = nn.epochErr;
%[err(1,ii), ~] = nntest(nn, test_x, test_y);
W(:,ii) = nn.W;
GAMMA(:,ii) = nn.gamma;
BETA(:,ii) = nn.beta;

beforeErr = [beforeErr; err_tmp];
end
nn.learningRate = init_lr;
nn.norm_learningRate = init_nlr;
%beforeErr = mean(beforeErr);
before = mean(err);
%DL end
%PCA start

save('saved_config_mnist_basic_training_cnn');

end