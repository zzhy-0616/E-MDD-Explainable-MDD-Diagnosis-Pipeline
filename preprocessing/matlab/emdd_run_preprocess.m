function emdd_run_preprocess(cfgPath)
%EMDD_RUN_PREPROCESS E-MDD MATLAB 预处理入口（由 Python workflow 调用）
% 读取 JSON 配置，按顺序执行：可选 ICA 脚本 → 6s 切 epoch → train/test 划分 → 可选 Z-norm
%
% 用法（由 workflow 自动调用）:
%   matlab -batch "addpath('.../preprocessing/matlab'); emdd_run_preprocess('runs/xxx/matlab_preprocess.json')"

cfgPath = char(cfgPath);
cfg = jsondecode(fileread(cfgPath));

repoRoot = cfg.repo_root;
addpath(fullfile(repoRoot, 'preprocessing', 'matlab'));

if isfield(cfg, 'eeglab_init') && ~isempty(cfg.eeglab_init)
    eval(cfg.eeglab_init);
elseif exist('eeglab', 'file')
    eeglab nogui;
end

fprintf('=== E-MDD MATLAB preprocess start ===\n');

if isfield(cfg, 'ica') && cfg.ica.enabled
    if ~isfield(cfg.ica, 'script') || isempty(cfg.ica.script)
        error('ica.enabled=true 但未配置 ica.script');
    end
    fprintf('[step] ICA / artifact removal: %s\n', cfg.ica.script);
    run(cfg.ica.script);
end

if cfg.epoch_6s.enabled
    fprintf('[step] 6s epoch slicing\n');
    emdd_epoch_6s(cfg.epoch_6s);
end

if cfg.split.enabled
    fprintf('[step] subject-level train/test split\n');
    emdd_split_train_test(cfg.split);
end

if cfg.znorm.enabled
    fprintf('[step] channel Z-norm (train stats only)\n');
    emdd_znorm(cfg.znorm);
end

fprintf('=== E-MDD MATLAB preprocess done ===\n');
end
