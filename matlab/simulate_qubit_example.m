function simulate_qubit_example(idx_ex, out_dir, K)
%SIMULATE_QUBIT_EXAMPLE Simulate one driven single-qubit response example.
%
% The qubit is driven through x/y Gaussian control channels and subject to
% coloured classical dephasing noise coupled through sigma_z. Six Pauli
% eigenstate preparations are propagated for each stochastic trajectory.
% Final X/Y/Z expectations are ensemble averaged and saved.
%
% Output variables
%   pulse_parameters : [2 x 10 x 3]
%   data             : [1 x 18]

    rng(idx_ex, 'twister');

    T_phys = 0.25e-6;
    M = 1000;

    t_grid = linspace(0, T_phys, M);
    n_steps = M - 1;
    dt = T_phys / n_steps;
    t_mid = 0.5 * (t_grid(1:end-1) + t_grid(2:end));

    % Pauli operators
    sx = [0 1; 1 0];
    sy = [0 -1i; 1i 0];
    sz = [1 0; 0 -1];

    % Rotating-frame drift
    omega_q = 2*pi*5.33e9;
    omega_d = 2*pi*5.30e9;
    delta_q = omega_q - omega_d;
    H_static = 0.5 * delta_q * sz;

    % Gaussian pulse family
    n_max = 10;
    n_ctrl = 2;
    omega1 = 2;

    spacing = T_phys / (n_max + 1);
    centres = ((1:n_max).') * spacing;
    min_sigma = 0.04 * spacing;
    max_sigma = 0.30 * spacing;
    std_gauss = min_sigma + (max_sigma - min_sigma) * rand;
    A_max = pi / (sqrt(2*pi) * std_gauss);

    % pulse_parameters(control, gaussian, :) = [A_k, tau_k, sigma]
    pulse_parameters = zeros(n_ctrl, n_max, 3);

    for idx_op = 1:n_ctrl
        amp = A_max * (2*rand(n_max, 1) - 1);
        pulse_parameters(idx_op,:,1) = amp;
        pulse_parameters(idx_op,:,2) = centres;
        pulse_parameters(idx_op,:,3) = std_gauss;
    end

    gaussian = @(idx,t) sum( ...
        squeeze(pulse_parameters(idx,:,1)) .* ...
        exp(-0.5 * ((t - squeeze(pulse_parameters(idx,:,2))) ./ ...
        pulse_parameters(idx,1,3)).^2) ...
    );

    u_x = (omega1/2) * arrayfun(@(tm) gaussian(1, tm), t_mid);
    u_y = -(omega1/2) * arrayfun(@(tm) gaussian(2, tm), t_mid);

    % Final observables
    observables = {sx, sy, sz};
    n_obs = numel(observables);

    % Pauli eigenprojectors in order +X, -X, +Y, -Y, +Z, -Z
    paulis = {sx, sy, sz};
    rho_init = cell(6,1);
    init_idx = 1;

    for p = 1:3
        [V, D] = eig(paulis{p});
        eigenvalues = real(diag(D));
        [~, pos] = max(eigenvalues);
        [~, neg] = min(eigenvalues);

        v_plus = V(:,pos);
        v_minus = V(:,neg);

        rho_init{init_idx} = v_plus * v_plus';
        rho_init{init_idx + 1} = v_minus * v_minus';
        init_idx = init_idx + 2;
    end

    n_init = numel(rho_init);
    d = 2;

    Y_init = zeros(d^2, n_init, 'like', 1 + 1i);
    for ii = 1:n_init
        Y_init(:,ii) = reshape(rho_init{ii}, d^2, 1);
    end

    % Coloured classical dephasing noise
    g1 = 50;
    H_beta = g1 * sz;

    f_pos = (0:(M/2 - 1)).' / T_phys;
    S_z = zeros(size(f_pos));
    nonzero = f_pos > 0;
    S_z(nonzero) = (1e9) ./ f_pos(nonzero) + (1e-9) * f_pos(nonzero);
    S_z(1) = S_z(2);

    beta = generate_arbitrary_noise(S_z, T_phys, M, K);
    noise_time = squeeze(beta(1,:,:,1));
    beta_mid = 0.5 * (noise_time(:,1:end-1) + noise_time(:,2:end));

    % Liouvillian pieces under column-major vectorisation
    I = speye(d);
    liouvillian = @(H) -1i * (kron(I, H) - kron(H.', I));

    L0 = liouvillian(H_static);
    Lx = liouvillian(sx);
    Ly = liouvillian(sy);
    Lb = liouvillian(H_beta);

    data_accum = zeros(1, n_init*n_obs);

    for r = 1:K
        Y = Y_init;

        for k = 1:n_steps
            L = L0 + u_x(k)*Lx + u_y(k)*Ly + beta_mid(r,k)*Lb;
            Y = expm(full(L) * dt) * Y;
        end

        final_expect = zeros(n_init, n_obs);

        for ii = 1:n_init
            rho_final = reshape(Y(:,ii), d, d);

            for obs_idx = 1:n_obs
                final_expect(ii, obs_idx) = real( ...
                    trace(observables{obs_idx} * rho_final) ...
                );
            end
        end

        data_accum = data_accum + reshape(final_expect.', 1, []);
    end

    data = data_accum / K;

    pulse_parameters = single(pulse_parameters);
    data = single(data);

    fname = fullfile(out_dir, sprintf('dataset_%04d.mat', idx_ex));
    save(fname, 'pulse_parameters', 'data');

    fprintf('Saved %s (%d final expectations)\n', fname, numel(data));
end
