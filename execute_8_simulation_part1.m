

gen_size_param = [2,4,6,8,10];
layer_param = [1];



for iii = layer_param
    for jjj = gen_size_param
        parfor idx11 = 1:50 
           % DNN_crossValidation_batch_norm_n_class_itern(j,i); 
           saveFile = strcat(num2str(1),'_reFA_DNN_batch_all_k_fold_','train_num_',num2str(100),'layer_',num2str(iii),'k_fold',num2str(1), '_gen_size', num2str(jjj),'_idx',num2str(idx11),'/_regenerate_',num2str(jjj),'.mat');
            if exist(saveFile, 'file')== 2 
                
                continue;
            else
            end
            Paper_figure_DNN_BC_WS_basic_training_no_eval(jjj,iii,idx11);
            Paper_figure_8_DNN_BC_WS_simulation(jjj,iii,idx11);
            
        end
    end
end

