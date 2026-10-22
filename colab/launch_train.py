"""Launch a detached ketos fine-tuning run on the Colab VM; poll /content/<RUN>/train.log afterwards.

Edit RUN / ARGS before `colab exec -f`; the CLI transmits the file content, so no argparse here.
"""
import subprocess
from pathlib import Path

RUN = 'ft_v1'
DATA = '/content/data_v1'
# ppocrv6 trains with a cosine schedule, which needs a fixed epoch budget (early stopping -> inf steps).
# Lines are padded to max_width 2560 px: batch 16 OOMs on a 22 GB L4, so 4 x 4 accumulation.
ARGS = ['-B', '4', '--accumulate-grad-batches', '4', '-r', '1e-4', '--augment', '-q', 'fixed', '-N', '50', '-p', '0.9', '-u', 'NFD',
        '--resize', 'union']

out = Path('/content') / RUN
out.mkdir(parents=True, exist_ok=True)
xmls = sorted(str(p) for p in Path(DATA).glob('*.xml'))
cmd = ['ketos', '-d', 'cuda:0', '-s', '42', '--workers', '2', 'train', '-f', 'page',
       '-i', '/content/models/medium.safetensors', '-o', f'{out}/model', *ARGS, *xmls]
(out / 'cmd.txt').write_text(' '.join(cmd))
log = open(out / 'train.log', 'w')
subprocess.Popen(['nohup', *cmd], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print('launched', RUN, len(xmls), 'files')
