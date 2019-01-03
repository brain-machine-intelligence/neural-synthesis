%% figure 4
lay = 6;
gen_idx = 1;
gen_size = 2;

for class_num = 1:10
    fitplotN_tot{class_num} = zeros(2,250);
end

for gen = gen_size 
    for idx = 1:50
        file_str= strcat('.\1_reFA_DNN_batch_all_k_fold_train_num_100layer_',num2str(lay),'k_fold1_gen_size', num2str(gen),'_idx',num2str(idx),'\_regenerate_',num2str(gen),'.mat' );
        load(file_str);
        for class_num = 1 :10
           fitplotN_tot{class_num}(idx,:) = fitplotN{1,class_num}(1,:);
        end
        
    end
end


num_epoch = 250;
avg_num = 50;

figure;hold on;
color_c = ['-r', '-g', '-b', 'c', 'k', 'p', 'bl', 'gr', 'w', 'y'];
tot_Line = [];

for idx = 1:10
    
    %H(start_idx)   = shadedErrorBar(num_epoch*(idx) : num_epoch*(idx+1)-1, squeeze(error_default(idx,:,:)), {@mean, @(x) 0.5*std(x)},'lineprops','-c');
    %H(start_idx+1) = shadedErrorBar(num_epoch*(idx) : num_epoch*(idx+1)-1, squeeze(error_regen(idx,:,:)), {@mean, @(x) 0.5*std(x)},'lineprops','-b');
    %H(start_idx+2) = shadedErrorBar(num_epoch*(idx) : num_epoch*(idx+1)-1, squeeze(error_random(idx,:,:)), {@mean, @(x) 0.5*std(x)},'lineprops', '-g');
    %H(start_idx+3) = shadedErrorBar(num_epoch*(idx) : num_epoch*(idx+1)-1, squeeze(error_GAN(idx,:,:)), {@mean, @(x) 0.5*std(x)},'lineprops', '-k');
    Line = shadedErrorBar(1:num_epoch,fitplotN_tot{idx}(:,:), {@mean, @(x) std(x)/sqrt(avg_num)},'lineprops', color_c(idx));
    tot_Line = [tot_Line Line];
end

first_argu = [];
for idx = 1:10
    first_argu = [first_argu tot_Line(idx).mainLine];
end

%legend(first_argu, ...
%    'Class1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Class9','Class10');


xlabel('Generation','fontsize',17,'fontname','arial');
ylabel('Fitness Value','fontsize',17,'fontname','arial');