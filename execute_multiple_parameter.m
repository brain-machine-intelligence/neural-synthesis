

gen_size_param = [2,4,6,8,10];
layer_param = [1,3,6];


for i = layer_param
    for j = gen_size_param
        
        DNN_crossValidation_batch_norm_n_class_itern(j,i); 
    end
end

