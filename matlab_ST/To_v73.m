% 保存先を上書きせず、_v73.mat に変換して保存する
input_dir = './Training/';
output_dir = './Training_v73/';

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

files = dir(fullfile(input_dir, '*.mat'));

for k = 1:length(files)
    input_file = fullfile(input_dir, files(k).name);
    [~, name, ~] = fileparts(files(k).name);
    output_file = fullfile(output_dir, [name, '.mat']);  % 拡張子そのまま

    try
        data = load(input_file);



        
        save(output_file, '-struct', 'data', '-v7.3');
        fprintf('Converted: %s -> %s\n', input_file, output_file);
    catch ME
        fprintf('❌ Failed to convert: %s\n   Reason: %s\n', input_file, ME.message);
    end
end
