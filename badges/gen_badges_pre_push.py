from pylint.lint import Run
import glob
import anybadge
import pytest
import sys
import os 

scripts_path = glob.glob('cloud-detection-code/scripts/*.py')

pylint_files = []
for f in scripts_path:
    pylint_files.append(f)

results = Run(pylint_files, do_exit=False)
# `exit` is deprecated, use `do_exit` instead
pylint_overall = results.linter.stats['global_note']

# Define thresholds: <2=red, <4=orange <8=yellow <10=green
thresholds = {2: 'red',
              4: 'orange',
              6: 'yellow',
              10: 'green'}

badge = anybadge.Badge('pylint', round(pylint_overall, 2), thresholds=thresholds)

badge.write_badge('pylint.svg', overwrite=True)

retcode = pytest.main(["cloud-detection-code/tests", "--ignore", "cloud-detection-code/tests/ci_ignore"])

if retcode == 0:
    test_badge = anybadge.Badge('tests', 'passing', default_color='green')
    test_badge.write_badge('pytest.svg', overwrite=True)
else:
    test_badge = anybadge.Badge('tests', 'failing', default_color='red')
    test_badge.write_badge('pytest.svg', overwrite=True)
