function denoised_ECG = denoising_rjmartis(ECG)
    [C, L] = wavedec(ECG, 9, 'db6');
    remains = sum(L(1:8));
    nc = C;
    nc(remains+1:end) = 0;
    denoised_ECG = waverec(nc, L, 'db6');
end