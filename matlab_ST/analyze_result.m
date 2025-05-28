clear
close all

addpath(genpath('matlab_ST'));
load('\\wsl.localhost\ubuntu\home\shint\QRNN3D\results\data\qrnn3d.mat')


HSI_restored01 = normalize01(HSI_restored);

val_mpsnr = calc_MPSNR(HSI_restored01, HSI_clean);
val_mssim = calc_MSSIM(HSI_restored01, HSI_clean);

fprintf("MPSNR: %#.4g, MSSIM: %#.4g\n", val_mpsnr, val_mssim);


implay(cat(2, HSI_clean, HSI_noisy, HSI_restored01))

