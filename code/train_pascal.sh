mkdir ./results/
mkdir ./results/PASCAL/
cd ./code/
matlab -nodisplay -r "run train_vgg_cedn_pascal_contour.m; exit"
cd ..
