mkdir ./results/
mkdir ./results/BSDS500/
cd ./code/
matlab -nodisplay -r "run train_vgg_cedn_pascal_contour_bsds.m; exit"
cd ..
