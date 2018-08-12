close all
clear
clc
rng(15)
nsample = 50;
X = rand(nsample, 2);
Y = zeros(nsample,1);
for i = 1:nsample
    y1 = X(i,1)+.2;
    y2 = X(i,1)+.1;
    if X(i,2)<= y1 && X(i,2)>= y2
        Y(i) = 0;
    elseif X(i,2)> y1
        Y(i) = -1;
    else
        Y(i) = 1;
    end
end
indexx = find(Y~=0);
X = X(indexx,:);
Y = Y(indexx,:);
nsample = size(X,1);
T = 10;
X = [X';ones(1,nsample)];
Y = Y';

% init weigth vector
w=[.01 .01 0]';
% figure
pos = find(Y'>0);
neg = find(Y'<0);
% figure
% M
movie_iter = 0;
counter = 0;
frameno=10;
for iteration = 1 : T  %<- in practice, use some stopping criterion!
    for ii = 1 : size(X,2)         %cycle through training set
        clf;
        axis square;
        plot(X(1,pos),X(2,pos),'go','Markersize', 10)
        hold on
        plot(X(1,neg),X(2,neg),'bx','Markersize', 10)
        minB = - (w(1) * 0 + w(3)) ./ w(2);
        maxB = - (w(1) * 1 + w(3)) ./ w(2);
        H = line([0, 1], [minB, maxB]);
        hold on;
        set(H, 'LineStyle', '-.', 'Linewidth', 3, 'Color', 'k')
        axis([0 1 0 1]);
        axis square;
        drawnow
        M0=getframe(gcf);
        flagg = 0;
        if sign(w'*X(:,ii)) ~= Y(ii) %wrong decision?
            w = w + 1*X(:,ii) * Y(ii);   %then add (or subtract) this point to w
            plot(X(1,ii),X(2,ii),'rs','Markersize', 10,'MarkerFaceColor','r')
            flagg = 1;
            counter = counter + 1;
        end
        axis([0 1 0 1]);
        axis square
        drawnow
        hold off
        if flagg == 1 && counter>=9
            
            for l = 1:frameno
                movie_iter = movie_iter+1;
                M1(movie_iter) = M0;
            end
            M1(movie_iter)=getframe(gcf);
            for l = 1:frameno
                movie_iter = movie_iter+1;
                M1(movie_iter) = M1(movie_iter -1);
            end
            
        end
    end
end
axis([0 1 0 1]);
axis square;
drawnow;
M1(movie_iter)=getframe(gcf);
for l = 1:frameno
    movie_iter = movie_iter+1;
    M1(movie_iter) = M1(movie_iter -1);
end

v = VideoWriter('learning_fast.avi');
open(v)
writeVideo(v,M1)
close(v)
% save data

