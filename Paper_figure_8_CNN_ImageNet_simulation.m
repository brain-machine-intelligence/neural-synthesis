function Paper_figure_8_CNN_ImageNet_simulation(gen_size_param, layer_num_param)
addpath('./NN','./util')
rep = 1;
%DL
if nargin < 2
    gen_size_param= 2;
    layer_num_param=1;
end


avr_num = 30; % 반복성능을 비교하기 위해서 
kFold = 1; % cross validation k의 숫자.
train_size = 32;
init_lr = 0.7;
init_nlr = 0.2;
learning_rate = init_lr;
batch_norm = 0;
kf = 1;
gen_size = gen_size_param;
layer_num = layer_num_param;

n_iter = 1;
opts.numepochs =  30;                %  Number of full sweeps through data


load('saved_config_mnist_basic_training_cnn');

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
        nn.range = 1;
        %폴더에 해서 자장하자.
        saveFile = strcat(num2str(rep),'_reFA_CNN_batch4_ITER_batch_all_k_fold_','train_num_',num2str(train_num),'layer_',num2str(layer),'k_fold',num2str(kf));
        nn.filename = strcat('.\\',saveFile,'\\');
        mkdir(nn.filename);
        root = nn.filename;
        
        save_fa_data = strcat(nn.filename, 'DL_ITER_FA_DATA');
        save(save_fa_data, 'v', 'nn','idx');
        
        %몇개를 생성할텐가
        nval = gen_size;
        dim_c = 50;
        best_w = zeros(nval,dim_c);
        %data_x = zeros(nval,784);
        new_xN = {};
        origin_xN = {};
        bestObjArrN = {};
        fitplotN = {};

        for class_num = 1:2
            new_xN{class_num} = zeros(nval,size(trainingFeatures,2));
            origin_xN{class_num} = zeros(nval,size(trainingFeatures,2));
            bestObjArrN{class_num} = zeros(nval,1);
            fitplotN{class_num} = zeros(nval,350);
        end

        for class_num = 1:2
            ValidationData = double(trainSetN_x{class_num}(1:end,:));
            size_train = size(ValidationData, 1);
            kk_t = randperm(size_train);
            candi_origin_xN = zeros(size_train, size(trainingFeatures,2));
            candi_bestObjArrN = zeros(size_train,1);
            candi_new_xN = zeros(size_train, size(trainingFeatures,2));
            
            for i=1:size_train

                x0 = ValidationData(kk_t(i),:);
                %origin_xN{class_num}(i,:) = x0;
                candi_origin_xN(i,:) = x0;
                [best, x0, bestObj, fitplotN{class_num}(i,:)] = pca_dl_fa_ndim_layer_normalize_iter(x0,i,layer+1, dim_c, save_fa_data);
                
                img = x0;
                best_w(i,:) = best;
                %new_xN{class_num}(i,:) = img;
                candi_new_xN(i,:) = img;
                %bestObjArrN{class_num}(i,:) = bestObj;
                candi_bestObjArrN(i,:) = bestObj;
                close all
            end
            
            [out, idx] = sort(candi_bestObjArrN,'descend');
            new_xN{class_num} = candi_new_xN(idx(1:nval),:);
            origin_xN{class_num} = candi_origin_xN(idx(1:nval),:);
        end
        n_img{it_num} = new_xN;
        o_img{it_num} = origin_xN;

        %[~, idx1] = sort(bestObjArr1,'descend');
        %[~, idx2] = sort(bestObjArr2,'descend');
        %[~, idx3] = sort(bestObjArr3,'descend');

        save TEST_VAL best_w new_xN %data_x idx1 idx2 idx3;

        x = gen_size;
        n = 2*x;
        if n > nval
           n = nval; 
        end
        saveFileName = strcat(saveFile,'_regenerate_',num2str(n));
        whole_new_dat = [];
        whole_new_out = [];
        for class_num = 1:2
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
        for class_num = 1: 2
            size_testN = size(testN_x{class_num}, 1);
            kk_nt = randperm(size_testN);

            test_new_dat = [test_new_dat; double(testN_x{class_num}(kk_nt(1:nval),:))];
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
        for class_num = 1: 2 
            
            size_trainN = size(trainSetN_x{class_num}, 1);
            kk_nt = randperm(size_trainN);

            train_new_dat = [train_new_dat; double(trainN_x{class_num}(kk_nt(1:nval),:))];
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
        for class_num = 1: 2
            
            size_trainN = size(trainSetN_x{class_num}, 1);
            kk_nt = randperm(size_trainN);
            random_noise = rand(size(trainN_x{class_num}(kk_nt(1:nval),:)))*0.2;
            
            train_new_dat = [train_new_dat; random_noise + double(trainN_x{class_num}(kk_nt(1:nval),:))];
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
        
%         
%         train_new_dat = [];
%         train_new_out = [];
%         for class_num = 1: 2 
%             
%             size_trainN = size(trainSetN_x{class_num}, 1);
%             kk_nt = randperm(size_trainN);
%             %random_noise = rand(size(trainN_x{class_num}(kk_nt(1:nval),:)))*0.2;
%             load(strcat('./GAN_generation/result_origin_img',int2str(class_num-1),'.mat'));
%             img_from_gan = [];
%             for idx_aug = 1:nval
%                 img_from_gan = [img_from_gan; double(reshape(squeeze(origin_img(idx_aug,:,:))',[1,784]))];
%             end
%             train_new_dat = [train_new_dat; img_from_gan];
%             train_new_out = [train_new_out; Train_N{class_num}(1:nval,:)];
%         end
% 
%         trainSet_x = [pre_GAN; train_new_dat;];
% 
%         pre_GAN = trainSet_x;
% 
%         trainSet_y = [pre_GAN_y; train_new_out;];
%         pre_GAN_y = trainSet_y;
%             %trainSet_y = [Train_1(1:train_num+n,:); Train_2(1:train_num+n,:)];
% 
% 
%         [trainSet_x, mu, tanh_opta] = zscore(trainSet_x);
%         test_x = normalize(preTest_x, mu, tanh_opta);
% 
%         ganErr = zeros(1,avr_num);
%         gan_nn_err = zeros(avr_num, opts.numepochs);
%         meanErr = [];
%         for ii = 1:avr_num
%             %rand('state',0)
%             ganNN = backup_GAN_nn;
%             [ganNN, ~] = nntrain(ganNN, trainSet_x, trainSet_y, opts, test_x, test_y);
%             [tmp_err, ganBad] = nntest(ganNN, test_x, test_y);
%             gan_nn_err(ii,:) = ganNN.epochErr;
%             ganErr(1,ii) = tmp_err;            
%             ganNN_err{1,ii} = tmp_err;
% 
%             meanErr = [meanErr; tmp_err];
%         end
%         backup_GAN_nn = ganNN;
% 
%         meanErr = [beforeErr meanErr];
%         ganErr = (meanErr)*100;
%         gan = mean(randErr)*100;
% 
%         error_GAN(it_num,:,:) = gan_nn_err(:,:);
        
    end
    

        filename = 'ScenarioResult.txt';
        %fname = strcat(nn.filename,filename);
        fn = sprintf('%s\\res_%s', root, filename); 
        a = fopen(fn,'w');
        fprintf(a,'before %f\nregenerate %f\nbasic %f\n',before,regenerate, basic);
        %makeImage(root);
        saveFileMat = strcat(saveFileName,'.mat');
        save(saveFileMat,'regenerateErr', 'basicErr', 'before', 'regenerate', 'basic', 'best_w','new_xN', 'bestObjArrN','basicErr', 'reErr', 'err','basicBad','regenerateBad','fitplotN', 'nn_err', 'denn_err','error_regen','error_basic', 'error_default', 'error_random', 'error_GAN', 'n_img', 'o_img');
        %save Scenario14_9 beforeErr regenerateErr basicErr afbad compbad;
        fclose('all');
     end
end