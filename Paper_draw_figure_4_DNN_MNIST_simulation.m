%% figure 4
load('.\1_reFA_DNN_batch_all_k_fold_train_num_100layer_6k_fold1_gen_size2\_regenerate_2.mat');

class_num = 9;
gan_file_name = strcat('.\GAN_generation\', 'result_origin_img', num2str(class_num), '.mat');

load(gan_file_name);

base_img = o_img{1}{1,class_num+1}(1,:);
default_img = o_img{1}{1,class_num+1}(1,:);
random_img = base_img + 0.2*rand(1,784);
gan_img = squeeze(origin_img(7,:,:))';
synthe_img = n_img{1}{1,class_num+1}(1,:);

reshaped_origin = reshape(base_img, [28, 28]);
reshaped_default = reshape(default_img, [28, 28]);
reshaped_random = reshape(random_img, [28, 28]);

reshaped_synthe = reshape(synthe_img, [28, 28]);


subplot(1,5,1);
imshow(reshaped_origin');

subplot(1,5,2);
imshow(reshaped_default');

subplot(1,5,3);
imshow(reshaped_random');

subplot(1,5,4);
imshow(gan_img');

subplot(1,5,5);
imshow(reshaped_synthe');



