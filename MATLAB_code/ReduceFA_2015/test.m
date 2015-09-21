wave = inputs(1,:);
[c, info] = fwt(wave, 'db6', 9);
remains = sum(info.Lc(1:6));
nc = c;
nc(remains+1:end) = 0;
new_wave = ifwt(nc, info);
plot(new_wave)
% hold
% plot(wave)