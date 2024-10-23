path_ortho = strcat('./polarimetry/data/B1/global_no_bg.png');
ortho = imread(path_ortho);

path_unregistered = strcat('./polarimetry/data/B1/moving.png');
unregistered = imread(path_unregistered);

[mp,fp] = cpselect(unregistered,ortho,Wait=true);

save('./polarimetry/output/B1/manual_selection.mat', 'mp', 'fp');