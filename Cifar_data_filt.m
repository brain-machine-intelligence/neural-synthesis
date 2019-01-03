
reTrainN_x = {};
reTestN_x = {};

for i = 1:5 
    
    load(strcat('cifar-10-batches-mat/data_batch_',int2str(i),'.mat'));
    tt = reshape(data, [size(data,1), 32,32,3]);
    n_data = tt;
    
    for class_num = 1:10
        if i == 1
            reTrainN_x{class_num} = [];
        end
        cl_idx = find(labels==class_num-1);
        reTrainN_x{class_num} = [reTrainN_x{class_num}; n_data(cl_idx,:,:,:)];
    end

end
% 
% for class_num = 1:10
%     reTrainN_x{class_num} = permute(reTrainN_x{class_num},[3,2,4,1]);
% end

load(strcat('cifar-10-batches-mat/test_batch.mat'));

tt = reshape(data, [size(data,1), 32,32,3]);
n_data = tt;

for class_num = 1:10
    reTestN_x{class_num} = [];
    
    cl_idx = find(labels==class_num-1);
    reTestN_x{class_num} = [reTestN_x{class_num}; n_data(cl_idx,:,:,:)];
end

% for class_num = 1:10
%     reTestN_x{class_num} = permute(reTestN_x{class_num},[3,2,4,1]);
% end
