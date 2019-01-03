function DNN_crossValidation_batch_norm_n_class_itern(gen_size_param, layer_num_param)
addpath('./NN','./util')
rep = 1;
%DL
if nargin < 2
    gen_size_param=6;
    layer_num_param=1;
end
load ReTrainData;
train_num = 100; % 각 클래스별로 몇개의 학습 데이터수를 가져갈것인지,
avr_num = 5; % 반복성능을 비교하기 위해서 
kFold = 1; % cross validation k의 숫자.

learning_rate = 0.7;
batch_norm = 1;

gen_size = gen_size_param;
layer_num = layer_num_param;

n_iter = 3;
opts.numepochs =  100;                %  Number of full sweeps through data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 어떤 클래스를 가져온것인가
trainN_x = {};

trainN_x{1} = reTrain0_x;
trainN_x{2} = reTrain1_x;
trainN_x{3} = reTrain2_x;
trainN_x{4} = reTrain3_x;
trainN_x{5} = reTrain4_x;
trainN_x{6} = reTrain5_x;
trainN_x{7} = reTrain6_x;
trainN_x{8} = reTrain7_x;
trainN_x{9} = reTrain8_x;
trainN_x{10} = reTrain9_x;

testN_x = {};

testN_x{1} = reTest0_x;
testN_x{2} = reTest1_x;
testN_x{3} = reTest2_x;
testN_x{4} = reTest3_x;
testN_x{5} = reTest4_x;
testN_x{6} = reTest5_x;
testN_x{7} = reTest6_x;
testN_x{8} = reTest7_x;
testN_x{9} = reTest8_x;
testN_x{10} = reTest9_x;

% 2개의 클래스의 정답.
Train_N{1} = [ones(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1)];
Train_N{2} = [zeros(60000,1) ones(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1)];
Train_N{3} = [zeros(60000,1) zeros(60000,1) ones(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1)];
Train_N{4} = [zeros(60000,1) zeros(60000,1) zeros(60000,1) ones(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1)];
Train_N{5} = [zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) ones(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1)];
Train_N{6} = [zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) ones(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1)];
Train_N{7} = [zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) ones(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1)];
Train_N{8} = [zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) ones(60000,1) zeros(60000,1) zeros(60000,1)];
Train_N{9} = [zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) ones(60000,1) zeros(60000,1)];
Train_N{10} = [zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) zeros(60000,1) ones(60000,1)];

%6000개의 데이터를 임의로 섞어 가져오기.


mean_of_train = 1;
% cross validation의 순서에 따라 실험.
for kf = kFold:-1: 1 % kf 테스트 데이터의 순번. kFold번째 -> kFold-1번째 -> ... 1번째
    
    trainSet_x = [];
    trainSetN_x = {};
       
    trainSet_y = [];
    
    testSet_x = [];
    testSet_y = [];
    
    for class_num = 1:10
        m = size(trainN_x{class_num}, 1);
        kk1 = randperm(m);
        trainSet_x = [trainSet_x; trainN_x{class_num}(kk1(1:train_num),:);];
        trainSet_y = [trainSet_y; Train_N{class_num}(1:train_num,:);];
        
        trainSetN_x{class_num} = trainN_x{class_num}(kk1(1:train_num),:);
        testSet_x = [testSet_x; testN_x{class_num}(:,:);];
        testSet_y = [testSet_y; Train_N{class_num}(1:size(testN_x{class_num},1),:);];
    end

    
    %batch normalization을 위한 0~1로 만들기
    train_x = double(trainSet_x) / 255;
    test_x  = double(testSet_x)  / 255;

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
nn = nnsetup([784 1000 800 600 400 200 100 10]);
nn.activation_function = 'tanh_opt';    %  tanh_optoid activation function
nn.learningRate = learning_rate;                %  tanh_opt require a lower learning rate
nn.batch_normalize = batch_norm;
opts.batchsize = 10;               %  Take a mean gradient step over this many samples 
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
%beforeErr = mean(beforeErr);
before = mean(err);
%DL end
%PCA start




%% layer and regenerated number
for xx = [layer_num]
    backup_regen_nn = nn;
    backup_basic_nn = nn;
    backup_default_nn = nn;
    backup_random_nn = nn;
    backup_GAN_nn = nn;
    
    error_default = zeros(n_iter, avr_num, opts.numepochs);
    error_regen = zeros(n_iter,avr_num,opts.numepochs);
    error_basic = zeros(n_iter,avr_num,opts.numepochs);
    error_random = zeros(n_iter,avr_num,opts.numepochs);
    error_GAN = zeros(n_iter, avr_num, opts.numepochs);
    
    
    pre_regen = pre_x;
    pre_regen_y = train_y;
    
    pre_basic = pre_x;
    pre_basic_y = train_y;
    
    pre_default = pre_x;
    pre_default_y = train_y;
    
    pre_random = pre_x;
    pre_random_y = train_y;
    
    pre_GAN = pre_x;
    pre_GAN_y = train_y;
    
    c = cov(pre_regen);
    [v, d] = eig(c);
    maxd = diag(d);
    [~, idx] = sort(maxd,'descend');
    n_img = {};
    o_img = {};
    for it_num = 1:n_iter
    
        layer = xx;

        %fit val에서의 c 값
        nn.cval = 0.1;
        %탐색 범위를 어디까지 할껀가
        nn.range = 0.2;
        %폴더에 해서 자장하자.
        saveFile = strcat(num2str(rep),'_reFA_DNN_ITER_batch_all_k_fold_','train_num_',num2str(train_num),'layer_',num2str(layer),'k_fold_',num2str(kf),'gen_size',num2str(gen_size));
        nn.filename = strcat('.\\',saveFile,'\\');
        mkdir(nn.filename);
        root = nn.filename;
        save_fa_data = strcat(nn.filename, 'DL_ITER_FA_DATA');
        save(save_fa_data, 'v', 'nn','idx');

        %몇개를 생성할텐가
        nval = gen_size;
        dim_c = 20;
        best_w = zeros(nval,dim_c);
        %data_x = zeros(nval,784);
        new_xN = {};
        origin_xN = {};
        bestObjArrN = {};
        fitplotN = {};

        for class_num = 1:10
            new_xN{class_num} = zeros(nval,784);
            origin_xN{class_num} = zeros(nval,784);
            bestObjArrN{class_num} = zeros(nval,1);
            fitplotN{class_num} = zeros(nval,250);
        end

        for class_num = 1:10
            ValidationData = double(trainSetN_x{class_num}(1:end,:))/255;
            size_train = size(ValidationData, 1);
            kk_t = randperm(size_train);

            for i=1:nval

                x0 = ValidationData(kk_t(i),:);
                origin_xN{class_num}(i,:) = x0;
                [best, x0, bestObj, fitplotN{class_num}(i,:)] = pca_dl_fa_ndim_layer_normalize_iter_DNN_batch(x0,i,layer+1, dim_c, save_fa_data);
                %data_x(i,:) = x0;
                %minT = min(x0);
                %table = x0 - minT;
                %maxT = max(table);
                %img = table/maxT;
                best_w(i,:) = best;
                img = x0;
                new_xN{class_num}(i,:) = img;
                
                bestObjArrN{class_num}(i,:) = bestObj;
                close all
            end
        end
        n_img{it_num} = new_xN;
        o_img{it_num} = origin_xN;

        %[~, idx1] = sort(bestObjArr1,'descend');
        %[~, idx2] = sort(bestObjArr2,'descend');
        %[~, idx3] = sort(bestObjArr3,'descend');

        %save TEST_VAL best_w new_xN %data_x idx1 idx2 idx3;

        x = gen_size;
        n = 2*x;
        if n > nval
           n = nval; 
        end
        saveFileName = strcat(saveFile,'_regenerate_',num2str(n));
        whole_new_dat = [];
        whole_new_out = [];
        for class_num = 1:10
            whole_new_dat = [whole_new_dat; new_xN{class_num}(1:nval,:);];
            whole_new_out = [whole_new_out; Train_N{class_num}(1:nval,:)];
        end
        
        trainSet_x = [pre_regen; whole_new_dat];
        trainSet_y = [pre_regen_y; whole_new_out];

        pre_regen = trainSet_x;
        pre_regen_y = trainSet_y;

        % normalize
        [trainSet_x, mu, tanh_opta] = zscore(trainSet_x);
        test_x = normalize(preTest_x, mu, tanh_opta);
       %% ex1 vanilla neural net
        %rand('state',0)
        reErr = zeros(1,avr_num);
        renn_err = zeros(avr_num, opts.numepochs);
        meanErr = [];
        for ii = 1:avr_num
            regenerateNN = backup_regen_nn;
            [regenerateNN, ~] = nntrain(regenerateNN, trainSet_x, trainSet_y, opts, test_x, test_y);
            [tmp_err, regenerateBad] = nntest(regenerateNN, test_x, test_y);
            renn_err(ii,:) = regenerateNN.epochErr;
            reErr(1,ii) = tmp_err;
            regenerateNN_err{1,ii} = tmp_err;

            meanErr = [meanErr; tmp_err];
        end
        backup_regen_nn = regenerateNN;
        meanErr = [beforeErr meanErr];
        regenerateErr = (meanErr)* 100;

        regenerate = mean(reErr)*100;
        error_regen(it_num,:,:) = renn_err(:,:);
            %%
            %ValidationData1(kk(nval+1:nval+n,:),:);
            %ValidationData2(kk(nval+1:nval+n,:),:);

        test_new_dat = [];
        test_new_out = [];
        for class_num = 1: 10 
            size_testN = size(testN_x{class_num}, 1);
            kk_nt = randperm(size_testN);

            test_new_dat = [test_new_dat; double(testN_x{class_num}(kk_nt(1:nval),:))/255];
            test_new_out = [test_new_out; Train_N{class_num}(1:nval,:)];
        end

        trainSet_x = [pre_basic; test_new_dat;];

        pre_basic = trainSet_x;

        trainSet_y = [pre_basic_y; test_new_out;];
        pre_basic_y = trainSet_y;
            %trainSet_y = [Train_1(1:train_num+n,:); Train_2(1:train_num+n,:)];


        [trainSet_x, mu, tanh_opta] = zscore(trainSet_x);
        test_x = normalize(preTest_x, mu, tanh_opta);

        deErr = zeros(1,avr_num);
        bann_err = zeros(avr_num, opts.numepochs);
        meanErr = [];
        for ii = 1:avr_num
            %rand('state',0)
            basicNN = backup_basic_nn;
            [basicNN, ~] = nntrain(basicNN, trainSet_x, trainSet_y, opts, test_x, test_y);
            [tmp_err, basicBad] = nntest(basicNN, test_x, test_y);
            bann_err(ii,:) = basicNN.epochErr;
            deErr(1,ii) = tmp_err;            
            basicNN_err{1,ii} = tmp_err;

            meanErr = [meanErr; tmp_err];
        end
        backup_basic_nn = basicNN;

        meanErr = [beforeErr meanErr];
        basicErr = (meanErr)*100;
        basic = mean(deErr)*100;

        error_basic(it_num,:,:) = bann_err(:,:);
        
        
        train_new_dat = [];
        train_new_out = [];
        for class_num = 1: 10 
            
            size_trainN = size(trainSetN_x{class_num}, 1);
            kk_nt = randperm(size_trainN);

            train_new_dat = [train_new_dat; double(trainN_x{class_num}(kk_nt(1:nval),:))/255];
            train_new_out = [train_new_out; Train_N{class_num}(1:nval,:)];
        end

        trainSet_x = [pre_default; train_new_dat;];

        pre_default = trainSet_x;

        trainSet_y = [pre_default_y; train_new_out;];
        pre_default_y = trainSet_y;
            %trainSet_y = [Train_1(1:train_num+n,:); Train_2(1:train_num+n,:)];


        [trainSet_x, mu, tanh_opta] = zscore(trainSet_x);
        test_x = normalize(preTest_x, mu, tanh_opta);

        deErr = zeros(1,avr_num);
        denn_err = zeros(avr_num, opts.numepochs);
        meanErr = [];
        for ii = 1:avr_num
            %rand('state',0)
            defaultNN = backup_default_nn;
            [defaultNN, ~] = nntrain(defaultNN, trainSet_x, trainSet_y, opts, test_x, test_y);
            [tmp_err, defaultBad] = nntest(defaultNN, test_x, test_y);
            denn_err(ii,:) = defaultNN.epochErr;
            deErr(1,ii) = tmp_err;            
            defaultNN_err{1,ii} = tmp_err;

            meanErr = [meanErr; tmp_err];
        end
        backup_default_nn = defaultNN;

        meanErr = [beforeErr meanErr];
        defaultErr = (meanErr)*100;
        default = mean(deErr)*100;

        error_default(it_num,:,:) = denn_err(:,:);
        
        train_new_dat = [];
        train_new_out = [];
        for class_num = 1: 10 
            
            size_trainN = size(trainSetN_x{class_num}, 1);
            kk_nt = randperm(size_trainN);
            random_noise = rand(size(trainN_x{class_num}(kk_nt(1:nval),:)))*0.2;
            
            train_new_dat = [train_new_dat; random_noise + double(trainN_x{class_num}(kk_nt(1:nval),:))/255];
            train_new_out = [train_new_out; Train_N{class_num}(1:nval,:)];
        end

        trainSet_x = [pre_random; train_new_dat;];

        pre_random = trainSet_x;

        trainSet_y = [pre_random_y; train_new_out;];
        pre_random_y = trainSet_y;
            %trainSet_y = [Train_1(1:train_num+n,:); Train_2(1:train_num+n,:)];


        [trainSet_x, mu, tanh_opta] = zscore(trainSet_x);
        test_x = normalize(preTest_x, mu, tanh_opta);

        randErr = zeros(1,avr_num);
        randnn_err = zeros(avr_num, opts.numepochs);
        meanErr = [];
        for ii = 1:avr_num
            %rand('state',0)
            randomNN = backup_random_nn;
            [randomNN, ~] = nntrain(randomNN, trainSet_x, trainSet_y, opts, test_x, test_y);
            [tmp_err, randomBad] = nntest(randomNN, test_x, test_y);
            randnn_err(ii,:) = randomNN.epochErr;
            randErr(1,ii) = tmp_err;            
            randomNN_err{1,ii} = tmp_err;

            meanErr = [meanErr; tmp_err];
        end
        backup_random_nn = randomNN;

        meanErr = [beforeErr meanErr];
        randomErr = (meanErr)*100;
        random = mean(randErr)*100;

        error_random(it_num,:,:) = randnn_err(:,:);
        
        
        train_new_dat = [];
        train_new_out = [];
        for class_num = 1: 10 
            
            size_trainN = size(trainSetN_x{class_num}, 1);
            kk_nt = randperm(size_trainN);
            %random_noise = rand(size(trainN_x{class_num}(kk_nt(1:nval),:)))*0.2;
            load(strcat('./GAN_generation/result_origin_img',int2str(class_num-1),'.mat'));
            img_from_gan = [];
            for idx_aug = 1:nval
                img_from_gan = [img_from_gan; double(reshape(squeeze(origin_img(idx_aug,:,:))',[1,784]))];
            end
            train_new_dat = [train_new_dat; img_from_gan];
            train_new_out = [train_new_out; Train_N{class_num}(1:nval,:)];
        end

        trainSet_x = [pre_GAN; train_new_dat;];

        pre_GAN = trainSet_x;

        trainSet_y = [pre_GAN_y; train_new_out;];
        pre_GAN_y = trainSet_y;
            %trainSet_y = [Train_1(1:train_num+n,:); Train_2(1:train_num+n,:)];


        [trainSet_x, mu, tanh_opta] = zscore(trainSet_x);
        test_x = normalize(preTest_x, mu, tanh_opta);

        ganErr = zeros(1,avr_num);
        gan_nn_err = zeros(avr_num, opts.numepochs);
        meanErr = [];
        for ii = 1:avr_num
            %rand('state',0)
            ganNN = backup_GAN_nn;
            [ganNN, ~] = nntrain(ganNN, trainSet_x, trainSet_y, opts, test_x, test_y);
            [tmp_err, ganBad] = nntest(ganNN, test_x, test_y);
            gan_nn_err(ii,:) = ganNN.epochErr;
            ganErr(1,ii) = tmp_err;            
            ganNN_err{1,ii} = tmp_err;

            meanErr = [meanErr; tmp_err];
        end
        backup_GAN_nn = ganNN;

        meanErr = [beforeErr meanErr];
        ganErr = (meanErr)*100;
        gan = mean(randErr)*100;

        error_GAN(it_num,:,:) = gan_nn_err(:,:);
        
    end
    

        filename = 'ScenarioResult.txt';
        %fname = strcat(nn.filename,filename);
        fn = sprintf('%s\\res_%s', root, filename); 
        a = fopen(fn,'w');
        fprintf(a,'before %f\nregenerate %f\nbasic %f\n',before,regenerate, basic);
        %makeImage(root);
        
        saveFileMat = strcat(nn.filename, saveFileName,'.mat');
        save(saveFileMat,'regenerateErr', 'basicErr', 'before', 'regenerate', 'basic', 'best_w','new_xN', 'bestObjArrN','basicErr', 'reErr', 'err','basicBad','regenerateBad','fitplotN', 'nn_err', 'denn_err','error_regen','error_basic', 'error_default', 'error_random', 'error_GAN', 'n_img', 'o_img');
        %save Scenario14_9 beforeErr regenerateErr basicErr afbad compbad;
        fclose('all');
     end
 end
end


