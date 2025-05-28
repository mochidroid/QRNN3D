%% generate dataset
rng(0)

%% for Pavia
preprocess = @(x)(center_crop(normalize(x), 340, 340)); % Pavia


basedir = 'data';
input_dir = './data/PaviaU/';
% output_dir = './PaviaCentre/Testing_mixture/';
newdir = fullfile(basedir, ['Pavia_mixture_full']);

if ~exist(newdir, 'dir')
    mkdir(newdir);
end

fns = {'PaviaU.mat'};
sigmas = [10, 30, 50, 70];
generate_dataset_mixture(input_dir, fns, newdir, sigmas, 'paviaU', preprocess);

% generate_dataset_gst(input_dir, fns, output_dir, sigmas, 'rad', preprocess);
% generate_dataset_gs(input_dir, fns, output_dir, sigmas, 'rad', preprocess);