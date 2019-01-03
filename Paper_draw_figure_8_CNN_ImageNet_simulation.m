%% figure 8 grid

load('.\CNN_result_fig8\1_reFA_CNN_batch4_ITER_batch_all_k_fold_train_num_32layer_1k_fold_1gen_size2_regenerate_2.mat');

iter_num=1;
num_epoch = size(error_basic,3);
avg_num = size(error_basic,2);

figure;hold on;
H(1) = shadedErrorBar(1:num_epoch, nn_err(:,:), {@mean, @(x) 0.5*std(x)},'lineprops','-r');

sq_err_1 = [nn_err(:,end) squeeze(error_default(1,:,:))];
sq_err_2 = [nn_err(:,end) squeeze(error_regen(1,:,:))];
sq_err_3 = [nn_err(:,end) squeeze(error_random(1,:,:))];
sq_err_4 = [nn_err(:,end) squeeze(error_GAN(1,:,:))];

for idx = 1:iter_num
    start_idx = (idx-1)*4+2;
    if idx ==1
    H(start_idx)   = shadedErrorBar(num_epoch*(idx) : num_epoch*(idx+1), sq_err_1, {@mean, @(x) 0.5*std(x)},'lineprops','-c');
    H(start_idx+1) = shadedErrorBar(num_epoch*(idx) : num_epoch*(idx+1), sq_err_2, {@mean, @(x) 0.5*std(x)},'lineprops','-b');
    H(start_idx+2) = shadedErrorBar(num_epoch*(idx) : num_epoch*(idx+1), sq_err_3, {@mean, @(x) 0.5*std(x)},'lineprops', '-g');
    H(start_idx+3) = shadedErrorBar(num_epoch*(idx) : num_epoch*(idx+1), sq_err_4, {@mean, @(x) 0.5*std(x)},'lineprops', '-k');
    else
    H(start_idx)   = shadedErrorBar(num_epoch*(idx) : num_epoch*(idx+1)-1, squeeze(error_default(idx,:,:)), {@mean, @(x) 0.5*std(x)},'lineprops','-c');
    H(start_idx+1) = shadedErrorBar(num_epoch*(idx) : num_epoch*(idx+1)-1, squeeze(error_regen(idx,:,:)), {@mean, @(x) 0.5*std(x)},'lineprops','-b');
    H(start_idx+2) = shadedErrorBar(num_epoch*(idx) : num_epoch*(idx+1)-1, squeeze(error_random(idx,:,:)), {@mean, @(x) 0.5*std(x)},'lineprops', '-g');
    H(start_idx+3) = shadedErrorBar(num_epoch*(idx) : num_epoch*(idx+1)-1, squeeze(error_GAN(idx,:,:)), {@mean, @(x) 0.5*std(x)},'lineprops', '-k');
    
    end
end

 %legend([H(2).mainLine, H(3).mainLine, H(4).mainLine, H(5).mainLine], ...
 %   '\default', 'proposed', ...
 %   'random', 'GAN');



xlabel('# of epochs for training','fontsize',17,'fontname','arial');
ylabel('Error rate(%)','fontsize',17,'fontname','arial');

%% New type

load('.\CNN_result_fig8\1_reFA_CNN_batch4_ITER_batch_all_k_fold_train_num_32layer_1k_fold_1gen_size2_regenerate_2.mat');

iter_num=1;
num_epoch = size(error_basic,3);
avg_num = size(error_basic,2);

figure;hold on;
%H(1) = shadedErrorBar(1:num_epoch, nn_err(:,:), {@mean, @(x) 0.5*std(x)},'lineprops','-r');
error_default = 100*error_default;
error_regen = 100*error_regen;
error_random = 100*error_random;
error_GAN = 100*error_GAN;
nn_err = 100*nn_err;

sq_err_1 = [nn_err(:,end) squeeze(error_default(1,:,:))];
sq_err_2 = [nn_err(:,end) squeeze(error_regen(1,:,:))];
sq_err_3 = [nn_err(:,end) squeeze(error_random(1,:,:))];
sq_err_4 = [nn_err(:,end) squeeze(error_GAN(1,:,:))];

for idx = 1:iter_num
    start_idx = (idx-1)*4+2;
    if idx ==1
    H(start_idx)   = shadedErrorBar(num_epoch*(idx) : num_epoch*(idx+1), sq_err_1, {@mean, @(x) 0.5*std(x)},'lineprops','-c');
    H(start_idx+1) = shadedErrorBar(num_epoch*(idx) : num_epoch*(idx+1), sq_err_2, {@mean, @(x) 0.5*std(x)},'lineprops','-b');
    H(start_idx+2) = shadedErrorBar(num_epoch*(idx) : num_epoch*(idx+1), sq_err_3, {@mean, @(x) 0.5*std(x)},'lineprops', '-g');
    H(start_idx+3) = shadedErrorBar(num_epoch*(idx) : num_epoch*(idx+1), sq_err_4, {@mean, @(x) 0.5*std(x)},'lineprops', '-k');
    else
    H(start_idx)   = shadedErrorBar(num_epoch*(idx) : num_epoch*(idx+1)-1, squeeze(error_default(idx,:,:)), {@mean, @(x) 0.5*std(x)},'lineprops','-c');
    H(start_idx+1) = shadedErrorBar(num_epoch*(idx) : num_epoch*(idx+1)-1, squeeze(error_regen(idx,:,:)), {@mean, @(x) 0.5*std(x)},'lineprops','-b');
    H(start_idx+2) = shadedErrorBar(num_epoch*(idx) : num_epoch*(idx+1)-1, squeeze(error_random(idx,:,:)), {@mean, @(x) 0.5*std(x)},'lineprops', '-g');
    H(start_idx+3) = shadedErrorBar(num_epoch*(idx) : num_epoch*(idx+1)-1, squeeze(error_GAN(idx,:,:)), {@mean, @(x) 0.5*std(x)},'lineprops', '-k');
    
    end
end

 %legend([H(2).mainLine, H(3).mainLine, H(4).mainLine, H(5).mainLine], ...
 %   '\default', 'proposed', ...
 %   'random', 'GAN');



%xlabel('# of epochs for training','fontsize',17,'fontname','arial');
%ylabel('Error rate(%)','fontsize',17,'fontname','arial');
set(gcf,'color','w')
set(gca,'FontSize',15)
Y_MIN = 0.08;
Y_MAX = 0.15;
set(gca,'XTick',[100, 150, 200]);
set(gca,'YTick',[9 12 15]);

%% New type + error_basic

load('.\CNN_result_fig8\1_reFA_CNN_batch4_ITER_batch_all_k_fold_train_num_32layer_2k_fold1gen_size2_regenerate_2.mat');

iter_num=1;
num_epoch = size(error_basic,3);
avg_num = size(error_basic,2);
nn_err = 100*nn_err;
figure;hold on;
H(1) = shadedErrorBar(1:num_epoch, nn_err(:,:), {@mean, @(x) 0.5*std(x)},'lineprops','-m');
error_default = 100*error_default;
error_regen = 100*error_regen;
error_random = 100*error_random;
error_basic = 100*error_basic;
%nn_err = 100*nn_err;

sq_err_1 = [nn_err(:,end) squeeze(error_default(1,:,:))];
sq_err_2 = [nn_err(:,end) squeeze(error_regen(1,:,:))];
sq_err_3 = [nn_err(:,end) squeeze(error_random(1,:,:))];
sq_err_4 = [nn_err(:,end) squeeze(error_basic(1,:,:))];

for idx = 1:iter_num
    start_idx = (idx-1)*4+2;
    if idx ==1
    H(start_idx)   = shadedErrorBar(num_epoch*(idx) : num_epoch*(idx+1), sq_err_1, {@mean, @(x) 0.5*std(x)},'lineprops','-c');
    H(start_idx+1) = shadedErrorBar(num_epoch*(idx) : num_epoch*(idx+1), sq_err_2, {@mean, @(x) 0.5*std(x)},'lineprops','-b');
    H(start_idx+2) = shadedErrorBar(num_epoch*(idx) : num_epoch*(idx+1), sq_err_3, {@mean, @(x) 0.5*std(x)},'lineprops', '-g');
    H(start_idx+3) = shadedErrorBar(num_epoch*(idx) : num_epoch*(idx+1), sq_err_4, {@mean, @(x) 0.5*std(x)},'lineprops', '-r');
    
    else
    H(start_idx)   = shadedErrorBar(num_epoch*(idx) : num_epoch*(idx+1)-1, squeeze(error_default(idx,:,:)), {@mean, @(x) 0.5*std(x)},'lineprops','-c');
    H(start_idx+1) = shadedErrorBar(num_epoch*(idx) : num_epoch*(idx+1)-1, squeeze(error_regen(idx,:,:)), {@mean, @(x) 0.5*std(x)},'lineprops','-b');
    H(start_idx+2) = shadedErrorBar(num_epoch*(idx) : num_epoch*(idx+1)-1, squeeze(error_random(idx,:,:)), {@mean, @(x) 0.5*std(x)},'lineprops', '-g');
    H(start_idx+3) = shadedErrorBar(num_epoch*(idx) : num_epoch*(idx+1)-1, squeeze(error_error_basic(idx,:,:)), {@mean, @(x) 0.5*std(x)},'lineprops', '-r');
    
    end
end

 %legend([H(2).mainLine, H(3).mainLine, H(4).mainLine, H(5).mainLine], ...
 %   '\default', 'proposed', ...
 %   'random', 'GAN');



%xlabel('# of epochs for training','fontsize',17,'fontname','arial');
%ylabel('Error rate(%)','fontsize',17,'fontname','arial');
set(gcf,'color','w')
set(gca,'FontSize',15)
Y_MIN = 0.08;
Y_MAX = 0.15;
%ylim([8 15]);
%set(gca,'XTick',[0, 50,100, 150, 200]);
%set(gca,'YTick',[9 12 15]);

%% basic Training
load('.\CNN_result_fig8\1_reFA_CNN_batch4_ITER_batch_all_k_fold_train_num_32layer_1k_fold1gen_size2_regenerate_2.mat');

iter_num=1;
num_epoch = size(error_basic,3);
avg_num = size(error_basic,2);
nn_err = nn_err*100;
figure;hold on;
H(1) = shadedErrorBar(1:num_epoch, nn_err(:,:), {@mean, @(x) 0.5*std(x)},'lineprops','-r');

set(gcf,'color','w')
set(gca,'FontSize',15)
%Y_MIN = 0.08;
%Y_MAX = 0.15;
%ylim([8 15]);
set(gca,'XTick',[0, 50, 100]);
set(gca,'YTick',[0 10 25 50]);