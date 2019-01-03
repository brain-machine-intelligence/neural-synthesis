

gen_size_param = [2,4,6,8,10];
layer_param = [1,2];


for i = layer_param
    for j = gen_size_param
        for k = 1:10
           % DNN_crossValidation_batch_norm_n_class_itern(j,i); 
            Paper_figure_9_CNN_ImageNet_simulation(j, i, k);
        end
    end
end

