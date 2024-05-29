import os
import yaml
import subprocess


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def process_args(args):
    if not args.eval:
        os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)

    tasks = []
    data_cfgs = []
    for data_cfg_path in args.data_cfg_paths:
        with open(data_cfg_path, 'r') as f:
            data_cfg = yaml.load(f, Loader=yaml.FullLoader)
        data_cfgs.append(data_cfg)
        tasks.append(data_cfg['TYPE'])
    args.tasks = tasks
    args.data_cfgs = data_cfgs

    if args.eval_data_cfg_path:
        with open(args.eval_data_cfg_path, 'r') as f:
            eval_data_cfg = yaml.load(f, Loader=yaml.FullLoader)
        args.eval_data_cfg = eval_data_cfg
    else:
        args.eval_data_cfg = None
        
    return args