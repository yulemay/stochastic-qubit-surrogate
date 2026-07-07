function beta = generate_arbitrary_noise(P_desired, T, M, K)
%GENERATE_ARBITRARY_NOISE Generate coloured classical noise trajectories.
%
% The function preserves the random-phase spectral-synthesis structure used
% in the supplied research code.
%
% Inputs
%   P_desired : single-side-band spectral profile
%   T         : total simulation duration
%   M         : number of time samples
%   K         : number of stochastic realisations
%
% Output
%   beta      : [1 x K x M x 1]

    Ts = T/M;
    N = M;

    if mod(N, 2) ~= 0
        error('M must be even for this spectral construction.');
    end

    if numel(P_desired) ~= N/2
        error('P_desired must contain M/2 single-side-band samples.');
    end

    P_desired = P_desired(:);
    beta = zeros(1, K, M, 1);

    for idx_k = 1:K
        P_temp = sqrt(P_desired*N/Ts) .* ...
            exp(2*pi*1i*rand(N/2, 1));

        P_temp = [P_temp; flipud(conj(P_temp))];

        beta(1,idx_k,:,1) = real(ifft(P_temp));
    end
end
