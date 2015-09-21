function NoSigIndex = findNoSigWave(inputs)
    [m, n] = size(inputs);
    NoSigIndex = zeros(m,1);
    for i = 1:m
        wave = inputs(i,:);
        nosig = 0;
        for j = 1:(n-124)
            if ~any(wave(j:j+125-1))
                nosig = 1;
            end
        end
        NoSigIndex(i) = nosig;
    end
end