%% augment every data
clear;
file_root = '.\1_reFA_CNN_batch4_ITER_50_batch_all_k_fold_train_num_32layer_1k_fold1idx';
final_value_regen = zeros(12,1);
final_value_default = zeros(12,1);
final_value_basic = zeros(12,1);
final_value_random = zeros(12,1);
result_idx =1;
for idx = 1:6
    full_file_path = strcat(file_root, num2str(idx),'.\_regenerate_2.mat');
    load(full_file_path);
    final_value_regen(result_idx) = mean(error_regen(1,:,end));
    final_value_default(result_idx) = mean(error_default(1,:,end));
    final_value_basic(result_idx) = mean(error_basic(1,:,end));
    final_value_random(result_idx) = mean(error_random(1,:,end));
    result_idx = result_idx+1;
end

for idx = 11:16
    full_file_path = strcat(file_root, num2str(idx),'.\_regenerate_2.mat');
    load(full_file_path);
    final_value_regen(result_idx) = mean(error_regen(1,:,end));
    final_value_default(result_idx) = mean(error_default(1,:,end));
    final_value_basic(result_idx) = mean(error_basic(1,:,end));
    final_value_random(result_idx) = mean(error_random(1,:,end));
    result_idx = result_idx+1;
end

mean_regen = mean(final_value_regen);
mean_default = mean(final_value_default);
mean_basic = mean(final_value_basic);
mean_random = mean(final_value_random);

var_regen = var(final_value_regen);
var_default = var(final_value_default);
var_basic = var(final_value_basic);
var_random = var(final_value_random);
