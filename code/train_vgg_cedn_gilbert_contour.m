% train semantic segmentation with vggnet fcn
addpath(genpath('/home/drew/objectContourDetector/caffe-cedn/matlab'));
model_specs = sprintf('vgg-16-encoder-decoder-%s', 'contour-gilbert');
use_gpu = true;
model_file = sprintf('%s.prototxt', model_specs);
solver_file = sprintf('%s_solver.prototxt', model_specs);
param = struct('base_lr', 0.00001, 'lr_policy', 'fixed', 'weight_decay', 0.001, 'solver_type', 3, 'snapshot_prefix', sprintf('../models/GILBERT/%s',model_specs));
make_solver_file(solver_file, model_file, param);
mean_pix = [103.939, 116.779, 123.68];
matcaffe_fcn_vgg_init(use_gpu, solver_file, 0);
weights0 = caffe('get_weights');
vggnet = load('../models/VGG_ILSVRC_16_layers_fcn_model.mat');
for i=1:14, weights0(i).weights = vggnet.model(i).weights; end
caffe('set_weights', weights0);
caffe('set_phase_train');

imnames = textread('../data/GILBERT/train.txt', '%s');
length(imnames)

H = 224; W = 224;
mask = zeros(size(im))
fid = fopen(sprintf('../results/GILBERT/%s-w10-train-errors.txt', model_specs),'w');
for iter = 1 : 30
  tic
  loss_train = 0;
  error_train_mask = 0;
  error_train_contour = 0;
  rnd_idx = randperm(length(imnames));
  for i = 1:length(imnames),
    name = imnames{rnd_idx(i)};
    im = imread(['../data/GILBERT/' name '.jpg']);
    label = 0
    if name(-1)~='0',
      label = 1
    [ims, masks] = sample_image(im, mask);
    ims = ims(:,:,[3,2,1],:);
    for c = 1:3, ims(:,:,c,:) = ims(:,:,c,:) - mean_pix(c); end
    ims = permute(ims,[2,1,3,4]);
    output = caffe('forward', {ims});
    [loss_contour, delta_contour] = loss_crossentropy_paired_sigmoid_grad(output{1}, contours);
    delta_contour = reshape(single(delta_contour),[2,1,1,8]);
    caffe('backward', {delta_contour});
    caffe('update');
    loss_train = loss_train + loss_contour;
    contours_pred = output{1} > 0;
    error_train_contour = error_train_contour + sum(sum(sum(contours_pred~=contours)));
    if mod(i,200)==0,
      fprintf('Iter,i: %d,%d training error is %f with contour.\n', iter, i, loss_contour);
    end
  end
  error_train_contour  = error_train_contour / length(imnames);
  loss_train = loss_train / length(imnames);
  fprintf('Iter %d: training error is %f with contour in %f seconds.\n', iter, error_train_contour, toc);
  fprintf(fid, '%d %f\n', iter, error_train_contour);
  if mod(iter,5)==0,
    weights = caffe('get_weights');
    save(sprintf('../results/GILBERT/%s-w10_model_iter%03d.mat', model_specs, iter), 'weights');
  end
end
fclose(fid);
caffe('snapshot');
