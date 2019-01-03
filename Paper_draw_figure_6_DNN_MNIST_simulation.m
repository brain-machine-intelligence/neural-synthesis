

layer_num = [1, 3, 6];
gen_size = [2,4,6,8,10];

total_error_regen = zeros(5,50);
total_error_random = zeros(5,50);
total_error_basic = zeros(5,50);
total_error_default = zeros(5,50);
total_error_GAN = zeros(5,50);
tot_error_nn_err = zeros(5,50);

%var_ratio = 0.001;
%var_ratio = 1;

x = [2:2:10];
lay = 3;  % input parama


gen_idx = 1;
for gen = gen_size 
    for idx = 1:50
        file_str= strcat('.\1_reFA_DNN_batch_all_k_fold_train_num_100layer_',num2str(lay),'k_fold1_gen_size', num2str(gen),'_idx',num2str(idx),'\_regenerate_',num2str(gen),'.mat' );
        load(file_str);
        total_error_regen(gen_idx, idx) = error_regen(end)*100;
        total_error_random(gen_idx, idx) = error_random(end)*100;
        total_error_basic(gen_idx, idx) = error_basic(end)*100;
        total_error_default(gen_idx, idx) = error_default(end)*100;
        total_error_GAN(gen_idx, idx) = error_GAN(end)*100;
        tot_error_nn_err(gen_idx,idx) = nn_err(end)*100;
        
    end
    gen_idx = gen_idx+1;
end

total_error_regen = total_error_regen - tot_error_nn_err;
total_error_random = total_error_random - tot_error_nn_err;
total_error_basic = total_error_basic - tot_error_nn_err;
total_error_default = total_error_default - tot_error_nn_err;
total_error_GAN = total_error_GAN -tot_error_nn_err;

mean_regen_total = mean(total_error_regen,2);
std_regen_total = sqrt(sum((total_error_regen - repmat(mean_regen_total,[1 50])).^2,2));
sem_regen_total = std_regen_total./sqrt(50);

mean_random_total = mean(total_error_random,2);
std_random_total = sqrt(sum((total_error_random - repmat(mean_random_total,[1 50])).^2,2));
sem_random_total = std_random_total./sqrt(50);

mean_basic_total = mean(total_error_basic,2);
std_basic_total = sqrt(sum((total_error_basic - repmat(mean_basic_total,[1 50])).^2,2));
sem_basic_total = std_basic_total./sqrt(50);

mean_GAN_total = mean(total_error_GAN,2);
std_GAN_total = sqrt(sum((total_error_GAN - repmat(mean_GAN_total,[1 50])).^2,2));
sem_GAN_total = std_GAN_total./sqrt(50);

mean_default_total = mean(total_error_default,2);
std_default_total = sqrt(sum((total_error_default - repmat(mean_default_total,[1 50])).^2,2));
sem_default_total = std_default_total./sqrt(50);

scatter(x,mean_regen_total,'b');
hold on;
plot(x, mean_regen_total,'b');
hold on;
errorbar(x, mean_regen_total,sem_regen_total, '-b');
hold on;

scatter(x,mean_random_total,'g');
hold on;
plot(x, mean_random_total,'g');
hold on;
errorbar(x, mean_random_total,sem_random_total, '-g');

%scatter(x,mean_basic_total,'r');
%hold on;
%plot(x, mean_basic_total,'r');
%hold on;
%errorbar(x, mean_basic_total,sem_basic_total, '-r');

scatter(x,mean_GAN_total,'k');
hold on;
plot(x, mean_GAN_total,'k');
hold on;
errorbar(x, mean_GAN_total,sem_GAN_total, '-k');

scatter(x,mean_default_total,'c');
hold on;
plot(x, mean_default_total,'c');
hold on;
errorbar(x, mean_default_total,sem_default_total, '-c');


%[h, p] = ttest(total_error_regen', total_error_default');
%[h, p] = ttest(total_error_regen', total_error_basic');
%[h, p] = ttest(total_error_regen', total_error_GAN');
[h, p] = ttest(total_error_regen', total_error_random');

set(gcf,'color','w')
set(gca,'FontSize',15)
%Y_MIN = 0.08;
%Y_MAX = 0.15;
ylim([-7 3]);
xlim([0 11.5]);

set(gca,'XTick',[2, 4, 6, 8, 10]);
set(gca,'YTick',[-7  -5 -3 -1 1 3]);
