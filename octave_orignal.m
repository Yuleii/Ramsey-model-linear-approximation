% Problem Set 5, Question 2 g)
% Ramsey model with exogenous growth.
% Log utility and Cobb-Douglas production function.
% -------------------------------------------------

% House keeping.
clear all;

% Parameters.
alpha = 0.3;                                                                    % Curvature of production function.
beta  = 0.95;                                                                   % Discount factor.
delta = 0.10;                                                                   % Depreciation rate.
n = 0.01;                                                                       % Population growth.
g = 0.02;                                                                       % Harrod-Neutral technological growth.
sigma = 1;                                                                      % log utility

betatilde = beta*((1+g)*(1+n))^(1-sigma);                                       % normalized beta.
betatildetilde = betatilde/((1+g)*(1+n));

T = 50;                                                                         % Periods to be simulated.

% computing the steady state
% ==========================

% Normalized capital in the BGP.
kss_sharp = ( (1/betatildetilde - 1 + delta) / (alpha) )^(1/(alpha - 1));

% Normalized consumption in the BGP.
css_sharp =  kss_sharp^alpha - (1+g)*(1+n)*kss_sharp + (1-delta)*kss_sharp;


% getting entries of the A matrix,
% using steady state values and corresponding derivatives
% =======================================================
coeff1 = -(1.0/sigma)*css_sharp*alpha*(alpha-1.0)*...
    kss_sharp^(alpha - 2.0)*(1/((1+g)*(1+n)));

A = NaN*zeros(2,2);

A(1,1) = 1.0 + betatildetilde*coeff1;

A(1,2) = -coeff1;

A(2,1) = -1.0/((1+g)*(1+n));

A(2,2) = 1.0/(betatildetilde*(1+g)*(1+n));

[V,D] = eig(A); % returns eigenvectors collected in the columns of V,
                % and the corresponding eigenvalues on the diagonal of D

% finding out at which position the stable eigenvalue appears
for i_D = 1:2
    if abs(D(i_D,i_D)) < 1.0
       stab_col_ind = i_D;
    end   % of if
end % of for loop running over the diagonal of the matrix D


% Simulate model.
% ===============

% Initialize vectors to store variables in deviations.
x_sharp_dev = NaN*zeros(2,T);


% Initial conditions.
x_sharp_dev(2,1) = -0.10*kss_sharp;                                             % Initial normalized capital deviation: initial capital 10 percent below steady state level.
x_sharp_dev(1,1) = (V(1,stab_col_ind)/V(2,stab_col_ind))*x_sharp_dev(2,1);      % Initial normalized consumption deviation.

% Simulate.
for t = 2:T

    x_sharp_dev(2,t) = A(2,:)*x_sharp_dev(:,t-1);
    x_sharp_dev(1,t) = (V(1,stab_col_ind)/V(2,stab_col_ind))*x_sharp_dev(2,t);

end

% Obtain levels (normalized).
x_sharp_lev = x_sharp_dev + [css_sharp ; kss_sharp];

% Obtain levels.
X_lev = x_sharp_lev.*((1+n)*(1+g)).^(0:T-1);

% Obtain output.
Y_lev = x_sharp_lev(2,:).^alpha.*((1+n)*(1+g)).^(0:T-1);

% Merge capital, consumption and output levels.
X_lev = [X_lev; Y_lev];

% Capital-Output-ratio.
K_Y_ratio = X_lev(2,:)./X_lev(3,:);

% Plot results.
% =============

% Normalized consumption deviations against normalized capital deviations.
figure(1)
plot(x_sharp_dev(2,:),x_sharp_dev(1,:),'-x','LineWidth',3)
xlabel('capital dev. : k^{#}_t - k^{# *}','FontSize',20);
ylabel('consumption dev. : c^{#}_t - c^{# *}','FontSize',20)

% Consumption levels, capital levels, and output levels against time.
figure(2)
plot(0:(T-1),X_lev,'LineWidth',3)
xlabel('Time','FontSize',20);
legend('Consumption level C_t','Capital level K_t','Output level Y_t')

% Consumption levels against capital levels.
figure(3)
plot(X_lev(2,:),X_lev(1,:),'-x','LineWidth',3)
xlabel('Capital level K_t','FontSize',20);
ylabel('Consumption level C_t','FontSize',20)

% Capital levels against output levels.
figure(4)
plot(X_lev(2,:),X_lev(3,:),'-x','LineWidth',3)
xlabel('Capital level K_t','FontSize',20);
ylabel('Output level Y_t','FontSize',20)

% Capital-output-ratio over time.
figure(5)
plot(0:(T-1),K_Y_ratio,'LineWidth',3)
xlabel('Time','FontSize',20);
ylabel('Capital-Output-ratio K_t/Y_t','FontSize',20)