

layer_num = [1, 3, 6];
gen_size = [2,4,6,8,10];

total_error_regen = zeros(5,10);
total_error_random = zeros(5,10);
total_error_basic = zeros(5,10);
total_error_default = zeros(5,10);
total_error_GAN = zeros(5,10);

var_ratio = 0.01;
%var_ratio = 1;

x = [2:2:10];
lay = 3;
gen_idx = 1;
for gen = gen_size 
    for idx = 1:10
        file_str= strcat('.\1_reFA_DNN_batch_all_k_fold_train_num_50layer_',num2str(lay),'k_fold1_gen_size', num2str(gen),'_idx',num2str(idx),'\_regenerate_',num2str(gen),'.mat' );
        load(file_str);
        total_error_regen(gen_idx, idx) = error_regen(3,1,end)*100;
        total_error_random(gen_idx, idx) = error_random(3,1,end)*100;
        total_error_basic(gen_idx, idx) = error_basic(3,1,end)*100;
        total_error_default(gen_idx, idx) = error_default(3,1,end)*100;
        total_error_GAN(gen_idx, idx) = error_GAN(3,1,end)*100;
        
        
    end
    gen_idx = gen_idx+1;
end


mean_regen_total = mean(total_error_regen,2);
var_regen_total = sum((total_error_regen - repmat(mean_regen_total,[1 10])).^2,2) * var_ratio;

mean_random_total = mean(total_error_random,2);
var_random_total = sum((total_error_random - repmat(mean_random_total,[1 10])).^2,2) * var_ratio;

mean_basic_total = mean(total_error_basic,2);
var_basic_total = sum((total_error_basic - repmat(mean_basic_total,[1 10])).^2,2) * var_ratio;

mean_GAN_total = mean(total_error_GAN,2);
var_GAN_total = sum((total_error_GAN - repmat(mean_GAN_total,[1 10])).^2,2) * var_ratio;

mean_default_total = mean(total_error_default,2);
var_default_total = sum((total_error_default - repmat(mean_default_total,[1 10])).^2,2) * var_ratio;

scatter(x,mean_regen_total,'b');
hold on;
plot(x, mean_regen_total,'b');
hold on;
errorbar(x, mean_regen_total,var_regen_total, '-b');
hold on;

scatter(x,mean_random_total,'g');
hold on;
plot(x, mean_random_total,'g');
hold on;
errorbar(x, mean_random_total,var_random_total, '-g');

scatter(x,mean_basic_total,'r');
hold on;
plot(x, mean_basic_total,'r');
hold on;
errorbar(x, mean_basic_total,var_basic_total, '-r');

scatter(x,mean_GAN_total,'k');
hold on;
plot(x, mean_GAN_total,'k');
hold on;
errorbar(x, mean_GAN_total,var_GAN_total, '-k');

scatter(x,mean_default_total,'c');
hold on;
plot(x, mean_default_total,'c');
hold on;
errorbar(x, mean_default_total,var_default_total, '-c');


%[h, p] = ttest(total_error_regen', total_error_default');
%[h, p] = ttest(total_error_regen', total_error_basic');
%[h, p] = ttest(total_error_regen', total_error_GAN');
[h, p] = ttest(total_error_regen', total_error_random');

set(gcf,'color','w')
set(gca,'FontSize',15)
%Y_MIN = 0.08;
%Y_MAX = 0.15;
ylim([2 10]);
xlim([0 11.5]);

set(gca,'XTick',[2, 4, 6, 8, 10]);
set(gca,'YTick',[2 4 6 8 10]);
