function [G] = Movies2matrix()
filename = 'movie-features.csv';
delimiter = {',',' '};
startRow = 2;
% endRow = 2;
formatSpec = '%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%[^\n\r]';

fileID = fopen(filename,'r');

dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'EmptyValue' ,NaN,'HeaderLines', startRow-1, 'ReturnOnError', false);

fclose(fileID);

M = length(dataArray);
N = length(dataArray{1,1});

G = zeros(N, M-1);

for j = 1:M-1
    temp = dataArray{1,j};
    for i = 1:N
        
        G(i,j) = str2num(temp{i,1});
        
    end
end
G = G(:,2:end);
end