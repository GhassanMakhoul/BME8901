
function [] = modularity_pipeline(inpf, sub)
    addpath("/mnt/ernie_ghassan/datasets/action_modularity/derivatives/fmriprep/")
    addpath(genpath("/home/ghassan/MATLAB-Drive/Analysis of Brain Networks/BCT"))
    addpath("/home/ghassan/MATLAB-Drive/Analysis of Brain Networks/bme8901_data/")
    disp(char(sub))
    disp(append("Running on sub: ", sub, " ID"))
    disp("Loading Parcels!")
    % load parcels
    parc_dir='/home/ghassan/MATLAB-Drive/Analysis of Brain Networks/bme8901_data/';
    schaefer100 = "parcellations_MNI/Schaefer2018_100Parcels_17Networks_order_FSLMNI152_2mm.nii.gz";
    parcels = niftiread(parc_dir+schaefer100);
    parcels = reshape(parcels, [], 1);
    num_parcels = 100; %predetermined

    Vol_data = niftiread(inpf);
    [nx,ny,nz, t] = size(Vol_data);
    nvox = nx*ny*nz;
    Vol_data = reshape(Vol_data, nx*ny*nz, t);

    X = zeros(num_parcels,t); 
    %get avg time series per parcel
    for i=1:num_parcels
        inds = parcels == i;
        X(i,:) = mean(Vol_data(inds(1:nvox),:),1, 'omitnan');
    end
    %construct adjacency from parcels
    disp("Getting sig correlations")
    X = normalize(X, 2);
    [n, t] = size(X);
    W = significant_correlations(X, 0.01);

    %plot sig corr over all time
    fig = figure('visible','off');
    imagesc(W)
    axis square; colorbar
    xlabel('node')
    ylabel('node')
    title('Significant Correlation')
    saveas(fig, append(sub, 'sig_corr_alltpts.png'))
    W(isnan(W)) =0;
    close(fig)

    disp("Mulstiscale modularity")
    %Multiscale modularity
    gamma = 0.5:0.05:2.5;
    Mg = zeros(n, numel(gamma));
    for h = 1:numel(gamma)
        Mg(:, h) = consensus_community_louvain_with_finetuning(W, gamma(h));
    end

    fig = figure('visible','off');
    plot(max(Mg))
    xlabel("gamma values")
    ylabel("Modules")
    title("Resolution vs Module Num")
    saveas(fig, append(sub,'multiscale resolution.png'))
    close(fig)

    %see stability of NMI
    [~, nmi] = partition_distance(Mg);
    fig = figure('visible','off');
    imagesc(nmi==1)
    axis square
    saveas(fig, append(sub,'NMI.png'))
    close(fig)

    %time for dynamic modularity
    l=3; % dnumber of slices I want in each slice
    XX = mat2cell(X, 100, 52*ones(1, l));
    disp("Dynamic Modularity")
    Mw = zeros(n, 12);
    for h = 1:3
        W = significant_correlations(XX{h}, 0.01);
        W(W < 0) = 0;
        W(isnan(W)) = 0;
        Mw(:, h) = consensus_community_louvain_with_finetuning(W);
    end
    
    P = zeros(n);
    for i = 1:l
        P = P + (Mw(:, i)==Mw(:, i)');
    end
    P = P / l;
    
    M = mean(P);
    fig = figure('visible','off');
    imagesc(P)
    axis square
    saveas(fig, append(sub,'dynamic_consensus_modularity.png'))
    close(fig)
    disp("DONE!")
    exit;
end


function visualize_timeseries(X, fout)
% function to visualize timeseries and correlation matrices

figure,
tiledlayout(2, 1)

nexttile
imagesc(X)
xlabel('timepoints')
ylabel('brain regions')
colorbar

nexttile
imagesc(corr(X'), [-0.5 0.5])
xlabel('brain regions')
ylabel('brain regions')
axis square
colorbar

fig = gcf;
exportgraphics(fig, fout)
close(fig)
end

function W = significant_correlations(X, pval)

% compute original correlation
[n, t] = size(X);
W = corr(X');

s = 1000;
P = zeros(n, n, s);
for i = 1:s
    X0 = X;

    % shuffle each column separately
    for j = 1:t
        X0(randperm(n), j) = X(1:n, j);
    end

    % compute null correlation;
    P(:, :, i) = (abs(corr(X0')) >= abs(W));
end
P = mean(P, 3);

% threshold by p value
W(P >= pval) = 0;

end



function [m1, q1_next] = community_louvain_with_finetuning(W, gamma)

if ~exist('gamma', 'var')
    gamma = 1;
end

% number of nodes
n = size(W, 1);

% initial modularity vector
m1 = 1:n;

% initial modularity values
q1_prev = -inf;
q1_next = 0;

%%% begin iterative finetuning %%%
% while modularity increases
while q1_next - q1_prev > 1e-5

    % run modularity with previous affiliation
    q1_prev = q1_next;
    [m1, q1_next] = community_louvain(W, gamma, m1, 'negative_sym');
end
%%% end iterative finetuning %%%

end



function [M, Q] = consensus_community_louvain_with_finetuning(W, gamma)

if ~exist('gamma', 'var')
    gamma = 1;
end

n = size(W, 1);
k = 100;

P = W;
while 1
    M = zeros(n, k);
    Q = zeros(1, k);
    for i = 1:k
        [M(:, i), Q(i)] = community_louvain_with_finetuning(P, gamma);
    end

    P = zeros(n);
    for i = 1:k
        P = P + (M(:, i)==M(:, i)');
    end
    P = P / k;

    if all((P==0) | (P==1))
        break
    end
end
M = M(:, 1);
Q = Q(1);

end
