function Paper_figure_DNN_BC_WS_basic_training_no_eval(gen_size,layer_num, idx_file)
addpath('./NN','./util')
rep = 1;
%DL
if nargin < 2
    gen_size_param=2;
    layer_num_param=1;
end
train_num = 50; % 각 클래스별로 몇개의 학습 데이터수를 가져갈것인지,
load BC_WIS_DAT;
avr_num = 1; % 반복성능을 비교하기 위해서 

filt_Wiscon_data;
learning_rate = 0.7;
batch_norm = 1;

opts.numepochs =  100;                %  Number of full sweeps through data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 어떤 클래스를 가져온것인가
% 어떤 클래스를 가져온것인가
trainN_x = {};

trainN_x{1} = reTrain0_x;
trainN_x{2} = reTrain1_x;

testN_x = {};

testN_x{1} = reTest0_x;
testN_x{2} = reTest1_x;

% 2개의 클래스의 정답.
Train_N{1} = [ones(60000,1) zeros(60000,1)];
Train_N{2} = [zeros(60000,1) ones(60000,1)];
%6000개의 데이터를 임의로 섞어 가져오기.
%6000개의 데이터를 임의로 섞어 가져오기.


% cross validation의 순서에 따라 실험.
   
    trainSet_x = [];
    trainSetN_x = {};
       
    trainSet_y = [];
    
    testSet_x = [];
    testSet_y = [];
       
    for class_num = 1:2
        m = size(trainN_x{class_num}, 1);
        kk1 = randperm(m);
        trainSet_x = [trainSet_x; trainN_x{class_num}(kk1(1:train_num),:);];
        trainSet_y = [trainSet_y; Train_N{class_num}(1:train_num,:);];
        
        trainSetN_x{class_num} = trainN_x{class_num}(kk1(1:train_num),:);
        testSet_x = [testSet_x; testN_x{class_num}(:,:);];
        testSet_y = [testSet_y; Train_N{class_num}(1:size(testN_x{class_num},1),:);];
    end

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
    nn = nnsetup([30 50 40 20 2]);
    nn.activation_function = 'tanh_opt';    %  tanh_optoid activation function
    nn.learningRate = learning_rate;                %  tanh_opt require a lower learning rate
    nn.batch_normalize = batch_norm;
    opts.batchsize = 10;               %  Take a mean gradient step over this many samples 
    opts.validation = 1;
    nn.is_six = 1;
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
%beforeErr = mean(beforeErr);
before = mean(err);
%DL end
%PCA start


save(strcat('saved_bc_ws_basic_training', num2str(gen_size),'_',num2str(layer_num),'_', num2str(idx_file)));

end


