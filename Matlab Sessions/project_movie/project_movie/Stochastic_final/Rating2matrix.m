function [rating, idx_train, idx_test] = Rating2matrix()
filename = 'ratings-train.csv';
delimiter = ',';
startRow = 2;
formatSpec = '%f%f%f%[^\n\r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'EmptyValue' ,NaN,'HeaderLines' ,startRow-1, 'ReturnOnError', false);
fclose(fileID);
ratingstrain = [dataArray{1:end-1}];

filename = 'ratings-test.csv';
delimiter = ',';
startRow = 2;
formatSpec = '%f%f%f%[^\n\r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'EmptyValue' ,NaN,'HeaderLines' ,startRow-1, 'ReturnOnError', false);
fclose(fileID);
ratingstest = [dataArray{1:end-1}];

num_user = max(max(ratingstest(:,1)), max(ratingstrain(:,1)));
num_movie = max(max(ratingstest(:,2)), max(ratingstrain(:,2)));

rating = zeros(num_user,num_movie);
L_train = length(ratingstrain);
L_test = length(ratingstest);

idx_train = zeros(L_train,1);
idx_test = zeros(L_test,1);

for i = 1:L_train
    rating(ratingstrain(i,1), ratingstrain(i,2)) = ratingstrain(i,3);
    idx_train(i) = sub2ind([num_user,num_movie], ratingstrain(i,1), ratingstrain(i,2));     
end

for i = 1:L_test
    rating(ratingstest(i,1), ratingstest(i,2)) = ratingstest(i,3);
    idx_test(i) = sub2ind([num_user,num_movie], ratingstest(i,1), ratingstest(i,2));     
end

end



