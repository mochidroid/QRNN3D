%% generate dataset
rng(0)

% input_dir = './Testing/';
% output_dir = './Testing_gst/';
% output_dir = './Testing_gs/';

input_dir = './PaviaCentre/';
output_dir = './PaviaCentre/Testing_mixture/';

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

files = dir(fullfile(input_dir, '*.mat'));
fns = {files.name};

sz = 512;
preprocess = @(x)(center_crop(rot90(x), sz, sz));
% preprocess = @(x)(center_crop(normalized(x), 340, 340)); % Pavia

sigmas = [10, 30, 50, 70];


% generate_dataset_gst(input_dir, fns, output_dir, sigmas, 'rad', preprocess);
% generate_dataset_gs(input_dir, fns, output_dir, sigmas, 'rad', preprocess);


preprocess = @(x)(center_crop(normalized(x), 340, 340)); % Pavia

%% for Pavia
datadir = '/media/kaixuan/DATA/Papers/Code/Matlab/ImageRestoration/ITSReg/code of ITSReg MSI denoising/data/real/new';
newdir = fullfile(basedir, ['Pavia_mixture_full']);
fns = {'PaviaU.mat'};
sigmas = [10, 30, 50, 70];
generate_dataset_mixture(input_dir, fns, newdir, sigmas, 'hsi', preprocess);