function generate_dataset_gs(datadir, fns, newdir, sigmas, gt_key, preprocess)
%GENERATE_DATASET_gst Generate test data with Case 7 and 8 noise settings:
% Gaussian (random σ∈sigmas), Stripe (±0.25, 5%), Impulse (5%)
%
% Inputs:
% - datadir: input folder of .mat files
% - fns: cell array of filenames (e.g., {'img1.mat', 'img2.mat'})
% - newdir: output folder path
% - sigmas: vector of candidate σ values (e.g., [10,30,50,70])
% - gt_key: field name in .mat file (e.g., 'rad')
% - preprocess: function handle to preprocess input image

    % min_amount = 0.05;
    % max_amount = 0.05;
    impulse_ratio = 0.05;

    if ~exist(newdir, 'dir')
        mkdir(newdir)
    end

    for k = 1:length(fns)
        fn = fns{k};
        fprintf('generate data (%d/%d): %s\n', k, length(fns), fn);
        filepath = fullfile(datadir, fn);
        mat = load(filepath);
        gt = getfield(mat, gt_key);

        if exist('preprocess','var') && ~isempty(preprocess)
            gt = preprocess(gt);
        end

        gt = normalize(gt);
        [H, W, B] = size(gt);

        all_band = randperm(B);
        b = floor(B/3);

        %--- Gaussian noise: バンドごとにランダムにσを選択 ---%
        idx = randi(length(sigmas), B, 1);
        sigmaB = sigmas(idx);           % e.g. [10; 30; 50; ...]
        % disp(sigmaB');
        s = reshape(sigmaB, [1,1,B]);
        input = gt + (s/255) .* randn(size(gt));

        %--- Impulse noise ---%
        band_impulse = all_band(b+1:2*b);
        for i = 1:length(band_impulse)
            input(:,:,band_impulse(i)) = imnoise(input(:,:,band_impulse(i)), 'salt & pepper', impulse_ratio);
        end

        % save(fullfile(newdir, fn), 'gt', 'input', 'sigmaB', '-v7.3');
        save(fullfile(newdir, fn), 'gt', 'input', 'sigmaB');
    end
end
