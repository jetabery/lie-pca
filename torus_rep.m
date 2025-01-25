clear;
clc;

% rng(1);

d=3;
n=1000;

% Generate a random unitary matrix Q. (This method ensures Haar measure)
% A = (randn(d) + 1i*randn(d))/sqrt(2);
% [Q1,R] = qr(A);
% R_diag_angles = diag(diag(R)./abs(diag(R)));
% Q = Q1*R_diag_angles;
Q = eye(d);
Q = [ 0.56689951-0.08703072i, -0.00821467+0.66547876i, -0.38691909+0.28002633i;
     -0.37446802-0.1125242i,  0.23983754-0.18549333i, -0.86852536-0.02908408i;
      0.60894251-0.38386407i, -0.1958485-0.65328713i, -0.12797213+0.01788349i];


% a = randi([-6 6], 1, d);
a = [-1 -2 3];
% v = randn(d, 1) + 1i * randn(d, 1);
v = [1+1i; 2-3i; -1-4i];
v = v/norm(v);
t=(1:n)/n; % evenly spaced
% t = rand(1, n); % not evenly spaced
x=zeros(d, n);
for j=1:n
    x(:, j) = Q*diag(exp(2*pi*1i*a*t(j)))*Q'*v; 
end
x = x + randn(d, n)/100 + 1i*randn(d, n)/100; % add noise
% x = x(:, randperm(n)); % shuffle order of data points

% dists = squareform(pdist(x', @(u, V) sqrt(sum(abs(u-V).^2, 2))));
% epsilon = mean(min(dists+eye(n))); 
% % epsilon = min(min(dists+n*eye(n)));
% % epsilon=0.1;
% K = exp(-dists.^2/(epsilon));
% K = (K+K')/2;
% % Sinkhorn
% Dfull=eye(n);
% for iter = 1 : 10
%     D=diag(1./sqrt(sum(K)));
%     Dfull=Dfull*D;
%     K = D*K*D;
%     K=(K+K')/2;
% end
% weights = diag(Dfull / sum(sum(Dfull)));
% sum(weights)
% 
% [U,D]=eig(K);
% % 
% % plot(U(:,end-1),U(:,end-2), '.')
% % plot3(U(:, end-1), U(:,end-2), U(:,end-3), '.')
% % 
% % %%
% 
% 
% 
% angles = angle(U(:,end-1)+1i*U(:,end-2));
% [~, idx] = sort(angles);
% x = x(:, idx); % This should hopefully restore the ordering


% plot(x(1,:))


b = [];
P = {};
fft(x')
for b_test=-10:10
    Pbx = sum(exp(-2i*pi*b_test*((1:n)-1)/n) .* x, 2);
    if norm(Pbx) > 10
        Pbx
        b = [b b_test];
        P{b_test+11} = (norm(Pbx)^(-2))*(Pbx)*(Pbx)';
    end
end

x_start = sum(x(:, 1:5), 2)/5;
x_approx = zeros(d,n);
for j=1:n
    Gj = eye(d);
    for freq=b
        Gj = Gj + P{freq+11}*(exp(2i*pi*freq*t(j)) - 1);
    end
    x_approx(:, j) = Gj*x_start;
end

% disp(strcat('Detected Frequencies: ', mat2str(sort(b))))

plot3(imag(x(1,:)), imag(x(2,:)), imag(x(3,:)), '.')
hold on
plot3(imag(x_approx(1,:)), imag(x_approx(2,:)), imag(x_approx(3,:)))
hold off