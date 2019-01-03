

gen_size_param = [2,6,10];
layer_param = [1];



for iii = layer_param
    for jjj = gen_size_param
        for idx11 = 1:10 
           % DNN_crossValidation_batch_norm_n_class_itern(j,i); 
            
            % save File명 지정
            saveFile = strcat(num2str(1),'_reFA_DNN_batch_all_k_fold_','train_num_',num2str(100),'layer_',num2str(iii),'k_fold',num2str(1), '_gen_size', num2str(jjj),'_idx',num2str(idx11),'/_regenerate_',num2str(jjj),'.mat');
            if exist(saveFile, 'file')== 2 
                
                continue;
            else
            end
            
            Paper_figure_DNN_MNIST_basic_training(jjj,iii,idx11);  % 초기 훈련 진행
            Paper_figure_4_n_5_DNN_MNIST_simulation(jjj,iii,idx11); % 위 초기 훈련된 네트워크 기반으로 (random, synthezed, GAN ) 등을 활용한 추가 학습
            
        end
    end
end

