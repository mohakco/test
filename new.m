clc; clear; close all;

% Matrix basics
A = [1 2 3; 4 5 6];          % 2x3 matrix
v = [1; 2; 3];                % column vector
x = -5:0.01:5;                % range for smooth plots
x_d = -5:1:5;                 % discrete range for stem plots

% Element-wise (ALWAYS use these with arrays)
x.^2        % square each element
x.*2        % multiply each element
1./x        % divide 1 by each element  ← most common mistake: forgetting the dot

% Matrix info
size(A)         % [2 3]
size(A,1)       % 2 (rows)
size(A,2)       % 3 (cols)
length(v)       % 3 (max dimension)
sum(A(:))       % sum of ALL elements (flatten first)
sum(A,1)        % column sums
sum(A,2)        % row sums
inv(A)          % matrix inverse

x = -10:0.01:10;    % continuous
x_d = -10:1:10;     % discrete

%--- Gaussian: bell curve centered at c ---
mu_0  = exp(-x.^2 / 2);           % centered at 0
mu_5  = exp(-(x-5).^2 / 2);       % centered at 5
mu_n5 = exp(-(x+5).^2 / 2);       % centered at -5

%--- Cauchy: 1/(1+x^2) ---
A_x = 1 ./ (1 + x.^2);            % peaks at 1 when x=0

%--- Triangular ---
mu_tri = max(0, 1 - abs(x));       % peak at 0, zero at ±1

% Continuous plot
figure(1);
plot(x, mu_0, 'b', 'LineWidth', 2); hold on;
plot(x, mu_5, 'r', 'LineWidth', 2);
plot(x, A_x,  'g', 'LineWidth', 2); hold off;
legend('Gaussian-0','Gaussian-5','Cauchy');
title('Membership Functions'); xlabel('x'); ylabel('\mu(x)'); grid on;

% Discrete stem plot
figure(2);
mu_d = exp(-x_d.^2 / 2);          % discrete Gaussian
stem(x_d, mu_d, 'b', 'filled', 'LineWidth', 1.5);
title('Discrete Membership'); xlabel('x'); ylabel('\mu(x)');
ylim([0 1.2]); grid on;

%% ============================================================
%% 3. COMPLEMENT FUNCTIONS
%% ============================================================

a = 0:0.01:1;   % domain of complement is always [0,1]

%--- Three types ---
C_std   = 1 - a;                           % standard:   straight line
C_cos   = 0.5 * (1 + cos(pi * a));        % cosine:     S-curve
C_yager = (1 - a.^2).^(1/2);             % yager w=2:  quarter circle

figure(3);
plot(a, C_std,   'b',   'LineWidth', 2); hold on;
plot(a, C_cos,   'r--', 'LineWidth', 2);
plot(a, C_yager, 'g-.', 'LineWidth', 2); hold off;
legend('Standard 1-a', 'Cosine', 'Yager w=2');
title('Complement Functions'); xlabel('\mu'); ylabel('C(\mu)'); grid on;

%% ============================================================
%% 4. YAGER COMPLEMENT — MULTIPLE W VALUES
%%    Key concept: as w increases → approaches threshold complement
%% ============================================================

figure(4);
a = 0:0.01:1;
w_vals = [0.5, 1, 2, 5, 10, 50];
colors = {'b','r','g','m','k','c'};
hold on;
for i = 1:length(w_vals)
    w = w_vals(i);
    C = (1 - a.^w).^(1/w);
    % w=1   → same as standard (1-a)
    % w→inf → step function at a=0.5 (threshold complement)
    plot(a, C, colors{i}, 'LineWidth', 2, 'DisplayName', sprintf('w=%.1f',w));
end
hold off;
legend('Location','southwest');
title('Yager Complement for Different w');
xlabel('a'); ylabel('C(a)'); grid on;

%% ============================================================
%% 5. YAGER UNION — 3D SURFACE  ← HARD TOPIC
%%    Formula: U(a,b) = min(1, (a^w + b^w)^(1/w))
%%    As w→inf: approaches max(a,b) = standard union
%% ============================================================

[A, B] = meshgrid(0:0.05:1, 0:0.05:1);
% meshgrid creates two 2D grids covering every (a,b) combination
% A varies column-wise, B varies row-wise

figure(5);
w_union = [1, 2, 5, 10];
for idx = 1:4
    w = w_union(idx);
    U_yag = min(1, (A.^w + B.^w).^(1/w));
    % min(1,...) caps result at 1 since union can't exceed 1
    % For w=1: min(1, a+b) — probabilistic sum
    % For w=10: almost identical to max(a,b)

    subplot(2,2,idx);
    surf(A, B, U_yag, 'EdgeColor','none');
    % surf → 3D surface, EdgeColor none = no grid lines on surface
    title(sprintf('Yager Union w=%d', w));
    xlabel('a'); ylabel('b'); zlabel('U(a,b)');
    colorbar; view(45,30); grid on;
    % view(azimuth, elevation) = camera angle
end
sgtitle('Yager Union — 3D Surfaces');

%% ============================================================
%% 6. YAGER INTERSECTION — 3D SURFACE  ← HARD TOPIC
%%    Formula: I(a,b) = 1 - min(1, ((1-a)^w + (1-b)^w)^(1/w))
%%    As w→inf: approaches min(a,b) = standard intersection
%% ============================================================

figure(6);
w_inter = [1, 2, 5, 10];
for idx = 1:4
    w = w_inter(idx);
    I_yag = 1 - min(1, ((1-A).^w + (1-B).^w).^(1/w));
    % Built from complement of union of complements
    % 1-A and 1-B = complements of a and b
    % min(1,...) = Yager union of those complements
    % 1 - ... = final intersection

    subplot(2,2,idx);
    surf(A, B, I_yag, 'EdgeColor','none');
    title(sprintf('Yager Intersection w=%d', w));
    xlabel('a'); ylabel('b'); zlabel('I(a,b)');
    colorbar; view(45,30); grid on;
end
sgtitle('Yager Intersection — 3D Surfaces');

%% ============================================================
%% 7. UNION vs INTERSECTION — 1D SLICE COMPARISON  ← HARD TOPIC
%%    Fix b=0.5, vary a from 0 to 1
%%    Shows how Yager converges to standard as w increases
%% ============================================================

figure(7);
a_line = 0:0.01:1;
b_fix  = 0.5;          % fix one input to get a 1D slice

% --- Union slice ---
subplot(1,2,1);
hold on;
for w = [1, 2, 5, 10]
    U_slice = min(1, (a_line.^w + b_fix^w).^(1/w));
    % b_fix^w is scalar, a_line.^w is array — works via broadcasting
    plot(a_line, U_slice, 'LineWidth', 2, 'DisplayName', sprintf('w=%d',w));
end
% Standard union for comparison
plot(a_line, max(a_line, b_fix), 'k--', 'LineWidth', 2, 'DisplayName','max(a,b)');
hold off;
legend; grid on;
title(sprintf('Union Slice at b=%.1f', b_fix));
xlabel('a'); ylabel('U(a,b)');

% --- Intersection slice ---
subplot(1,2,2);
hold on;
for w = [1, 2, 5, 10]
    I_slice = 1 - min(1, ((1-a_line).^w + (1-b_fix)^w).^(1/w));
    plot(a_line, I_slice, 'LineWidth', 2, 'DisplayName', sprintf('w=%d',w));
end
plot(a_line, min(a_line, b_fix), 'k--', 'LineWidth', 2, 'DisplayName','min(a,b)');
hold off;
legend; grid on;
title(sprintf('Intersection Slice at b=%.1f', b_fix));
xlabel('a'); ylabel('I(a,b)');

sgtitle('Union & Intersection Slices — Yager vs Standard');

%% ============================================================
%% 8. APPLY COMPLEMENT TO MEMBERSHIP FUNCTION
%%    Shows complement of a real membership function, not just [0,1]
%% ============================================================

figure(8);
x_d  = -10:1:10;
mu_d = exp(-x_d.^2 / 2);    % original: "close to 0" membership
w    = 2;

% Apply each complement formula to mu_d values (not to x!)
c_std = 1 - mu_d;
c_cos = 0.5 * (1 + cos(pi * mu_d));
c_yag = (1 - mu_d.^w).^(1/w);

subplot(2,2,1); stem(x_d, mu_d,  'b','filled'); title('Original \mu(x)');  ylim([0 1.2]); grid on;
subplot(2,2,2); stem(x_d, c_std, 'r','filled'); title('Standard Complement'); ylim([0 1.2]); grid on;
subplot(2,2,3); stem(x_d, c_cos, 'g','filled'); title('Cosine Complement');   ylim([0 1.2]); grid on;
subplot(2,2,4); stem(x_d, c_yag, 'm','filled'); title('Yager w=2');           ylim([0 1.2]); grid on;
sgtitle('Complements Applied to Discrete Membership Function');

%% ============================================================
%% 9. COMBINED SUMMARY PLOT
%%    Complement + Union + Intersection together in one figure
%% ============================================================

figure(9);
a_line = 0:0.01:1;
b_fix  = 0.5;
w      = 2;

C_line = 1 - a_line;                                          % complement
U_line = min(1, (a_line.^w + b_fix^w).^(1/w));               % union
I_line = 1 - min(1,((1-a_line).^w+(1-b_fix)^w).^(1/w));      % intersection

subplot(1,3,1);
plot(a_line, C_line, 'b', 'LineWidth', 2);
title('Complement C(a)=1-a'); xlabel('a'); grid on;

subplot(1,3,2);
plot(a_line, U_line, 'r', 'LineWidth', 2); hold on;
plot(a_line, max(a_line,b_fix), 'r--', 'LineWidth', 2); hold off;
legend('Yager w=2','max(a,b)');
title(sprintf('Union at b=%.1f',b_fix)); xlabel('a'); grid on;

subplot(1,3,3);
plot(a_line, I_line, 'g', 'LineWidth', 2); hold on;
plot(a_line, min(a_line,b_fix), 'g--', 'LineWidth', 2); hold off;
legend('Yager w=2','min(a,b)');
title(sprintf('Intersection at b=%.1f',b_fix)); xlabel('a'); grid on;

sgtitle('Combined: Complement + Union + Intersection (w=2)');

clc; clear; close all;

%% DATASET
rng(42);
c1 = [randn(50,1)*0.3+5.0, randn(50,1)*0.3+3.4];
c2 = [randn(50,1)*0.4+5.9, randn(50,1)*0.3+2.8];
c3 = [randn(50,1)*0.5+6.6, randn(50,1)*0.3+3.0];
X  = [c1; c2; c3];  % 150x2

%% FCM
[centers, U, objFcn] = fcm(X, 3, [2.0, 200, 1e-5, 0]);
% fcm(data, numClusters, [fuzziness, maxIter, epsilon, display])

[~, labels] = max(U, [], 1);  % hard labels from max membership
labels = labels';

%% PRINT
fprintf('Centers:\n'); disp(centers)
fprintf('Objective: %.4f in %d iterations\n', objFcn(end), length(objFcn));

%% PLOT
colors = ['r','b','g'];
figure;

% Plot 1: Clusters
subplot(1,3,1);
hold on;
for c = 1:3
    scatter(X(labels==c,1), X(labels==c,2), 50, colors(c), 'filled');
    scatter(centers(c,1), centers(c,2), 200, colors(c), 'p', 'filled', 'MarkerEdgeColor','k');
end
title('FCM Clusters'); xlabel('F1'); ylabel('F2'); grid on; hold off;

% Plot 2: Membership Heatmap
subplot(1,3,2);
imagesc(U); colorbar; colormap('parula');
xlabel('Sample'); ylabel('Cluster');
title('Membership Heatmap');

% Plot 3: Convergence
subplot(1,3,3);
plot(objFcn, 'b-o', 'LineWidth', 2);
xlabel('Iteration'); ylabel('Objective');
title('Convergence'); grid on;

sgtitle('Fuzzy C-Means Clustering');

clc; clear; close all;

%% DATASET (2 classes only for binary SVM)
rng(42);
c1 = [randn(50,1)*0.3+5.0, randn(50,1)*0.3+3.4];
c2 = [randn(50,1)*0.4+5.9, randn(50,1)*0.3+2.8];
X  = [c1; c2];               % 100x2
y  = [-ones(50,1); ones(50,1)];  % labels: -1 and +1

%% FUZZY MEMBERSHIP
s = zeros(100,1);
for c = 1:2
    cls    = 2*c - 3;              % maps c=1→-1, c=2→+1
    idx    = (y == cls);
    Xc     = X(idx,:);
    center = mean(Xc);             % class centroid
    dists  = sqrt(sum((Xc-center).^2, 2));  % distance from center
    r      = max(dists);           % radius
    s(idx) = 1 - dists/(r+0.1);   % closer to center = higher membership
end
s = max(s,0.1);  % floor at 0.1 so no point gets zero weight

%% TRAIN
model = fitcsvm(X, y, ...
    'KernelFunction', 'rbf', ...   % non-linear boundary
    'BoxConstraint',  1.0,   ...   % regularization
    'Weights',        s,     ...   % fuzzy weights (KEY LINE)
    'Standardize',    true,  ...
    'KernelScale',    'auto');

%% EVALUATE
yPred = predict(model, X);
acc   = sum(yPred==y)/numel(y)*100;
cm    = confusionmat(y, yPred);
fprintf('Accuracy: %.2f%%\n', acc);
fprintf('Support Vectors: %d\n', size(model.SupportVectors,1));
fprintf('Confusion Matrix:\n'); disp(cm);

%% PLOT
figure;

% Plot 1: Membership scatter
subplot(1,3,1);
gscatter(X(:,1), X(:,2), y, 'rb', 'ox', 8); hold on;
scatter(X(:,1), X(:,2), 60, s, 'filled', 'MarkerEdgeColor','k');
colorbar; colormap(gca,'parula');
title('Fuzzy Memberships'); grid on;

% Plot 2: Decision boundary
subplot(1,3,2);
x1r = linspace(min(X(:,1))-0.5, max(X(:,1))+0.5, 200);
x2r = linspace(min(X(:,2))-0.5, max(X(:,2))+0.5, 200);
[xx1,xx2]  = meshgrid(x1r, x2r);
[~, scores] = predict(model, [xx1(:), xx2(:)]);
scoreGrid  = reshape(scores(:,2), size(xx1));
contourf(xx1,xx2,scoreGrid,30,'LineStyle','none');
colormap(gca,'jet'); colorbar; hold on;
contour(xx1,xx2,scoreGrid,[0 0],'k-','LineWidth',2); % boundary line
gscatter(X(:,1),X(:,2),y,'rb','ox',8);
title('Decision Boundary'); grid on;

% Plot 3: Membership histogram
subplot(1,3,3);
histogram(s(y==-1),12,'FaceColor','r','FaceAlpha',0.6); hold on;
histogram(s(y== 1),12,'FaceColor','b','FaceAlpha',0.6);
xlabel('Membership s'); ylabel('Count');
legend('Class -1','Class +1');
title('Membership Distribution'); grid on;

sgtitle(sprintf('Fuzzy SVM | Accuracy: %.2f%%', acc));
