import json

import pytest

from paper_tools.spec_validation_tools import do_load_spec, do_validate_metric, make_tools
from paper_tools.state import ReproState


@pytest.fixture
def session(tmp_path):
    spec = tmp_path / 'spec.yaml'
    spec.write_text('''name: sample
repo: https://example.com/paper
run:
  command: python evaluate.py
  gpu: false
metrics:
  accuracy:
    pattern: 'accuracy ([0-9.]+)'
    recorded: 88.88
    tolerance: 1.0
  auc:
    pattern: 'auc ([0-9.]+)'
    recorded: 0.5
    tolerance: 0.1
    tolerance_type: rel
  unrecorded:
    pattern: 'value ([0-9.]+)'
''')
    return ReproState(str(tmp_path / 'workspace')), str(spec)


def test_load_spec_and_restore_session(session):
    state, path = session
    result = do_load_spec(state, path)
    assert result['paper'] == 'sample'
    assert result['run_command'] == 'python evaluate.py'
    assert result['gpu'] is False
    assert result['metrics']['accuracy']['recorded'] == 88.88
    assert ReproState(state.workspace).spec.name == 'sample'


@pytest.mark.parametrize('metric,value,status', [
    ('accuracy', 89.40, 'PASS'),
    ('accuracy', 89.88, 'PASS'),
    ('accuracy', 90.0, 'FAIL'),
    ('accuracy', 87.0, 'FAIL'),
    ('auc', 0.54, 'PASS'),
    ('auc', 0.60, 'FAIL'),
    ('unrecorded', 1.0, 'NO_RECORD'),
])
def test_validation_and_persistence(session, metric, value, status):
    state, path = session
    do_load_spec(state, path)
    # The reproduction tool normally stores this measured value after a run.
    state.data['reproduced'][state.key(metric)] = {'value': value}
    result = do_validate_metric(state, metric)
    assert result['status'] == status
    assert result['reproduced'] == value
    restored = ReproState(state.workspace)
    assert restored.data['validations'][state.key(metric)] == result


def test_missing_and_unknown_metric(session):
    state, path = session
    do_load_spec(state, path)
    assert 'not been reproduced' in do_validate_metric(state, 'accuracy')['message']
    assert do_validate_metric(state, 'unknown')['status'] == 'ERROR'
    assert not state.data['validations']


def test_langchain_invocation(session):
    state, path = session
    tools = {tool.name: tool for tool in make_tools(state)}
    assert set(tools) == {'load_paper_spec', 'validate_metric'}
    loaded = json.loads(tools['load_paper_spec'].invoke({'spec_path': path}))
    assert loaded['paper'] == 'sample'
    state.data['reproduced'][state.key('accuracy')] = {'value': 89.4}
    validated = json.loads(tools['validate_metric'].invoke({'metric': 'accuracy'}))
    assert validated['status'] == 'PASS'
    assert validated['difference'] == 0.52


def test_invalid_spec_is_reported(session, tmp_path):
    state, path = session
    tool = make_tools(state)[0]
    missing = json.loads(tool.invoke({'spec_path': str(tmp_path / 'missing.yaml')}))
    assert missing['status'] == 'ERROR'
    bad = tmp_path / 'bad.yaml'
    bad.write_text('name: bad\nmetrics: {accuracy: {recorded: 1}}\n')
    invalid = json.loads(tool.invoke({'spec_path': str(bad)}))
    assert invalid['status'] == 'ERROR'
    assert state.data['spec_path'] is None
