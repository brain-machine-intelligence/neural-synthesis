function [ output_args ] = normalizeHak( input_args )
%NORMALIZEHAK 이 함수의 요약 설명 위치
%   자세한 설명 위치
% 음수와 양수로 가득찬 이 데이터를 0~1로 만드는 함수가 될것임.
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

