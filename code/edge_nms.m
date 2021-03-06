clear; clc;

path_to_pdollar = '/media/data_cifs/cluster_projects/edges';
path_to_input = '/media/data_cifs/cluster_projects/BSDS500/bench/data/c3_output_gru/test';
path_to_output = '/media/data_cifs/cluster_projects/BSDS500/bench/data/c3_output_gru/test_nms';

addpath(genpath(path_to_pdollar));
addpath(genpath('/home/vijay/src/toolbox-master'));
mkdir(path_to_output);

iids = dir(fullfile(path_to_input, '*.png'));
fprintf("%d", length(iids));
for i = 1:length(iids)
    edge = imread(fullfile(path_to_input, iids(i).name));
    edge = 1-single(edge)/255;

    [Ox, Oy] = gradient2(convTri(edge, 4));
    [Oxx, ~] = gradient2(Ox);
    [Oxy, Oyy] = gradient2(Oy);
    O = mod(atan(Oyy .* sign(-Oxy) ./ (Oxx + 1e-5)), pi);
    % 2 for BSDS500 and Multi-cue datasets, 4 for NYUD dataset
    edge = edgesNmsMex(edge, O, 2, 5, 1.01, 8);
    edge = 1 - edge;
    imwrite(edge, fullfile(path_to_output, [iids(i).name(1:end-4) '.png']));

end
