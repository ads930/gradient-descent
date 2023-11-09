function [x, iter] = gdsolve(H, b, tol, maxiter)
%Gradient Descent Algorithm
%H is a N x N symmetric positive-definite matrix
% b is a vector of length N
% tol and maxiter specify halting conditions
% x is the solution
% iter is the number of iterations
% tol = 10^-6
%iterate until ||r_k||_2/||b||_2 is less than tol
N = size(H);
N = N(1);
xk = zeros(N,1);%guess
rk = b - H*xk;
r = b - H*xk;
iter = 0;
delta = tol;
convergence = norm(r)/norm(b);
while convergence > delta
    
    alpha = (rk'*rk)/(rk'*H*rk);
    xk = xk + alpha*rk;
    rk = b - H*xk;
    iter = iter + 1;
    if iter == maxiter
        break;
    end
end
x = xk;


end