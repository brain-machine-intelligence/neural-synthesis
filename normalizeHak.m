function [ output_args ] = normalizeHak( input_args )
%NORMALIZEHAK �� �Լ��� ��� ���� ��ġ
%   �ڼ��� ���� ��ġ
% ������ ����� ������ �� �����͸� 0~1�� ����� �Լ��� �ɰ���.
    x0 = input_args;
    %x0(find(x0 > 1)) = 1;
    %x0(find(x0 < 0)) = 0;
    minT = min(x0);
    maxT = max(x0);
    
    img = (x0-minT)/(maxT-minT);
    %table = x0 - minT;
    %maxT = max(table);
    %img = table/maxT;
    output_args = img;
end

