function generate_dataset(n_examples, out_dir, K)
%GENERATE_DATASET Generate stochastic single-qubit simulation examples.
%
%   generate_dataset(1000, '../data/raw')
%   generate_dataset(10, '../data/raw', 10)
%
% Inputs
%   n_examples : number of pulse-response examples
%   out_dir    : directory for dataset_XXXX.mat files
%   K          : stochastic realisations per example (default 1000)

    arguments
        n_examples (1,1) double {mustBeInteger, mustBePositive}
        out_dir (1,:) char
        K (1,1) double {mustBeInteger, mustBePositive} = 1000
    end

    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    for idx_ex = 1:n_examples
        simulate_qubit_example(idx_ex, out_dir, K);

        if mod(idx_ex, 10) == 0 || idx_ex == n_examples
            fprintf('Completed %d / %d examples\n', idx_ex, n_examples);
        end
    end
end
