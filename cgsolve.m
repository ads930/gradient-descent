function [x, iter] = cgsolve(H, b, tol, maxiter)
%Conjugate Gradients
N = size(H);
N = N(1);
x0 = zeros(N,1); %guess
r = b - H*x0;
delta = tol;
d = r;
iter = 0;
convergence = norm(r)/norm(b);
x = x0;
for k = 0: maxiter
    
    alpha = (r'*r)/(d'*H*d);
    x = x + alpha*d;
    r_old = r;
    r = r - alpha*H*d;
    B = (r'*r)/(r_old'*r_old);
    d = r + B*d;
    iter = iter + 1;
    if convergence < delta || iter == maxiter
        break;
    end

end

end