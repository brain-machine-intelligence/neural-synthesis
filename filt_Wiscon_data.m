
%% normalize Data

dat_size = size(BC_wis_dat,1);
BC_wis_dat = (BC_wis_dat - repmat(min(BC_wis_dat),[dat_size,1]))./repmat((max(BC_wis_dat)-min(BC_wis_dat)),[dat_size,1]);



%% divide into class
idx = find(BC_wis_dat(:,1)==0);

class0 = BC_wis_dat(idx,2:end);

idx = find(BC_wis_dat(:,1)==1);

class1 = BC_wis_dat(idx,2:end);

kk = randperm(size(class0,1));

reTrain0_x = class0(kk(1:train_num),:);
reTest0_x = class0(kk(train_num+1:end),:);

kk = randperm(size(class1,1));

reTrain1_x = class1(kk(1:train_num),:);
reTest1_x = class1(kk(train_num+1:end),:);

