from datetime import datetime, timezone, timedelta
import importlib
from pathlib import Path


def test_monitor_does_not_show_old_or_unmatched_green():
    m=importlib.import_module('monitor')
    assert hasattr(m,'confirmed_status'), 'Monitor has no freshness check'
    now=datetime.now(timezone.utc)
    good={'status':'COMPLIANT','captured_at':now.isoformat(),'source_sha256':'new'}
    assert m.confirmed_status(good,{'source_sha256':'new'},now)=='COMPLIANT'
    assert m.confirmed_status(good,{'source_sha256':'different'},now)=='AWAITING_RESULT'
    assert m.confirmed_status(good,{'source_sha256':'new'},now+timedelta(seconds=16))=='UNKNOWN'
