
%% New type + error_basic + 10 times

%load('.\1_reFA_DNN22_batch_all_k_fold_train_num_100layer_1k_fold1_gen_size2\_regenerate_2.mat');
tot_error_basic = zeros(10,100);
tot_error_default = zeros(10,100);
tot_error_GAN = zeros(10,100);
tot_error_random = zeros(10,100);
tot_error_regen = zeros(10,100);
tot_error_nn_err = zeros(10,100);

for idx = 1:10
    folder_name = strcat('.\1_reFA_DNN_batch_all_k_fold_train_num_100layer_6k_fold1_gen_size10_idx', num2str(idx),'\_regenerate_10.mat');
    load(folder_name);
    a=1;
    tot_error_basic(idx,:) = error_basic(1,1,:);
    tot_error_default(idx,:) = error_default(1,1,:);
    tot_error_GAN(idx,:) = error_GAN(1,1,:);
    tot_error_random(idx,:) = error_random(1,1,:);
    tot_error_regen(idx,:) = error_regen(1,1,:);
    tot_error_nn_err(idx,:) = nn_err(1,:);
    
end

iter_num=1;
num_epoch = 100;
avg_num = 10;

figure;hold on;
%H(1) = shadedErrorBar(1:num_epoch, nn_err(:,:), {@mean, @(x) 0.5*std(x)},'lineprops','-r');
error_basic = 100*tot_error_basic;
error_default = 100*tot_error_default;
error_regen = 100*tot_error_regen;
error_random = 100*tot_error_random;
error_GAN = 100*tot_error_GAN;
error_before = 100*tot_error_nn_err;

criteria1 = error_before(:,end);
criteria20 = repmat(criteria1,[1 20]);
criteria = repmat(criteria1,[1 100]);

sq_err_0 = error_before(:,end-19:end) - criteria20;
sq_err_1 = [criteria1-criteria1 error_default(:,:) - criteria] ;
sq_err_2 = [criteria1-criteria1 error_regen(:,:) - criteria] ;
sq_err_3 = [criteria1-criteria1 error_random(:,:) - criteria];
sq_err_4 = [criteria1-criteria1 error_GAN(:,:)- criteria];
sq_err_5 = [criteria1-criteria1 error_basic(:,:) - criteria];

H(1) = shadedErrorBar(1 : 20, sq_err_0, {@mean, @(x) 0.5*std(x)},'lineprops','-p');

for idx = 1:iter_num
    start_idx = (idx-1)*5+2;
    if idx ==1
    H(start_idx)   = shadedErrorBar(21 : 20+101, sq_err_1, {@mean, @(x) std(x)/sqrt(avg_num)},'lineprops','-c');
    H(start_idx+1) = shadedErrorBar(21 : 20+101, sq_err_2, {@mean, @(x) std(x)/sqrt(avg_num)},'lineprops','-b');
    H(start_idx+2) = shadedErrorBar(21 : 20+101, sq_err_3, {@mean, @(x) std(x)/sqrt(avg_num)},'lineprops', '-g');
    H(start_idx+3) = shadedErrorBar(21 : 20+101, sq_err_4, {@mean, @(x) std(x)/sqrt(avg_num)},'lineprops', '-k');
    %H(start_idx+4) = shadedErrorBar(21 : 20+101, sq_err_5, {@mean, @(x) std(x)/sqrt(avg_num)},'lineprops', '-r');
    
    else
    H(start_idx)   = shadedErrorBar(num_epoch*(idx) : num_epoch*(idx+1)-1, squeeze(error_default(idx,:,:)), {@mean, @(x) std(x)/sqrt(avg_num)},'lineprops','-c');
    H(start_idx+1) = shadedErrorBar(num_epoch*(idx) : num_epoch*(idx+1)-1, squeeze(error_regen(idx,:,:)), {@mean, @(x) std(x)/sqrt(avg_num)},'lineprops','-b');
    H(start_idx+2) = shadedErrorBar(num_epoch*(idx) : num_epoch*(idx+1)-1, squeeze(error_random(idx,:,:)), {@mean, @(x) std(x)/sqrt(avg_num)},'lineprops', '-g');
    H(start_idx+3) = shadedErrorBar(num_epoch*(idx) : num_epoch*(idx+1)-1, squeeze(error_GAN(idx,:,:)), {@mean, @(x) std(x)/sqrt(avg_num)},'lineprops', '-k');
    %H(start_idx+4) = shadedErrorBar(num_epoch*(idx) : num_epoch*(idx+1), squeeze(error_basic(idx,:,:)), {@mean, @(x) std(x)/sqrt(avg_num)},'lineprops', '-r');
    
    end
end

 %legend([H(2).mainLine, H(3).mainLine, H(4).mainLine, H(5).mainLine], ...
 %   '\default', 'proposed', ...
 %   'random', 'GAN');



%xlabel('# of epochs for training','fontsize',17,'fontname','arial');
%ylabel('Error rate(%)','fontsize',17,'fontname','arial');
set(gcf,'color','w');
set(gca,'FontSize',15);
Y_MIN = 0.0;
Y_MAX = -0.3;
ylim([-5.5 3]);
xlim([1 120]);
set(gca,'XTick',[200, 300, 400]);
set(gca,'YTick',[-5 -4 -3 -2 -1 0 1 2 3]);

%% basic Training for every layer, gen_num
layer_num = [1, 3, 6];
gen_size = [2,6,10];

total_error_nn = [];
for lay = layer_num 
   for gen = gen_size 
       for idx = 1:10
           file_str= strcat('.\1_reFA_DNN_batch_all_k_fold_train_num_100layer_',num2str(lay),'k_fold1_gen_size', num2str(gen),'_idx',num2str(idx),'\_regenerate_',num2str(gen),'.mat' );
           load(file_str);
           total_error_nn = [total_error_nn; nn_err];
       end
   end
end

iter_num=1;
error_basic = squeeze(total_error_nn);
num_epoch = size(error_basic,2);
avg_num = size(error_basic,1);
error_basic = error_basic*100;
figure;hold on;
H(1) = shadedErrorBar(1:num_epoch, error_basic(:,:), {@mean, @(x) std(x)/sqrt(avg_num)},'lineprops','-p');

set(gcf,'color','w')
set(gca,'FontSize',15)
%Y_MIN = 0.08;
%Y_MAX = 0.15;
ylim([10 60]);
set(gca,'XTick',[0, 50, 100]);
set(gca,'YTick',[0 10 25 50]);

