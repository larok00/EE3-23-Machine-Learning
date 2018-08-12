close all
clear
%clc
rng(15)

data = load('features.train');
y = data(:,1);
X = data(:,2:size(data,2));
dataT=load('features.test');
yT=dataT(:,1);
XT=dataT(:,2:size(dataT,2));
n=size(X,1);
nT=size(XT,1);
xmin1=min(min(X(:,1)),min(XT(:,1)));
xmin2=min(min(X(:,2)),min(XT(:,2)));
xmax1=max(max(X(:,1)),max(XT(:,1)));
xmax2=max(max(X(:,2)),max(XT(:,2)));
X=2*(X-[xmin1*ones(n,1),xmin2*ones(n,1)])*diag([1/(xmax1-xmin1),1/(xmax2-xmin2)])-1;
XT=2*(XT-[xmin1*ones(nT,1),xmin2*ones(nT,1)])*diag([1/(xmax1-xmin1),1/(xmax2-xmin2)])-1;
X=[X,ones(n,1)];
XT=[XT,ones(nT,1)];

ind=y==1|y==5;
indT=yT==1|yT==5;


% init weigth vector
w=[0 0 0]';
eta=0.1;
% figure

X = X(ind,:);
y = y(ind);

XT = XT(indT,:);
yT = yT(indT);

pos = find(y==1);
neg = find(y==5);
posT = find(yT==1);
negT = find(yT==5);

y(pos) = 1;
y(neg) = -1;
yT(pos) = 1;
yT(neg) = -1;

% figure
% M
movie_iter = 0;
counter = 0;
T=5000;
axis square;
plot(X(pos,1),X(pos,2),'go','Markersize', 10)
hold on
plot(X(neg,1),X(neg,2),'bx','Markersize', 10)
w_record = 10000*ones(100000,4);
for iteration = 1 : T  %<- in practice, use some stopping criterion!
    for ii = 1 : size(X,1)         %cycle through training set
        flagg = 0;
        if sign(X(ii,:)*w) ~= y(ii) %wrong decision?
            w = w + eta*X(ii,:)' * y(ii);   %then add (or subtract) this point to w
%             plot(X(ii,1),X(ii,2),'rs','Markersize', 10,'MarkerFaceColor','r')
            flagg = 1;
            counter = counter + 1;
            error = sum(sign(X*w) ~= y);
            w_record(counter,:) = [w', error];
        end
    end
end

w_last = w;
% axis square;
% plot(X(pos,1),X(pos,2),'go','Markersize', 10)
% hold on
% plot(X(neg,1),X(neg,2),'bx','Markersize', 10)
[a, b] = min(w_record(:,4));
w = w_record(b,1:3);
minB = - (w(1) * (-1) + w(3)) ./ w(2);
maxB = - (w(1) * 1 + w(3)) ./ w(2);
H = line([-1, 1], [minB, maxB]);
hold on;
set(H, 'LineStyle', '-.', 'Linewidth', 3, 'Color', 'k')
axis([-1 1 -1 1]);
axis square;
w = w_last;
minB = - (w(1) * (-1) + w(3)) ./ w(2);
maxB = - (w(1) * 1 + w(3)) ./ w(2);
H = line([-1, 1], [minB, maxB]);
hold on;
set(H, 'LineStyle', '-.', 'Linewidth', 3, 'Color', 'r')
drawnow

