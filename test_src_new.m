%clear;
%load('1_reFA_CNN_batch4_ITER_batch_all_k_fold_train_num_32layer_1k_fold2_regenerate_2.mat'); 

% figure;
% subplot(3,1,1)
% plot(o_img{1}{1,1}(2,:))
% subplot(3,1,2)
% plot(o_img{1}{1,2}(2,:))
% subplot(3,1,3)
% plot(o_img{1}{1,1}(2,:) - o_img{1}{1,2}(2,:))
% 
% figure;
% subplot(3,1,1)
% plot(n_img{1}{1,1}(2,:))
% subplot(3,1,2)
% plot(n_img{1}{1,2}(2,:))
% subplot(3,1,3)
% plot(n_img{1}{1,1}(2,:) - n_img{1}{1,2}(2,:))

figure;
subplot(3,1,1)
plot(o_img{1}{1,1}(2,:))
subplot(3,1,2)
plot(n_img{1}{1,1}(2,:))
subplot(3,1,3)
plot(n_img{1}{1,1}(2,:) - o_img{1}{1,1}(2,:))
