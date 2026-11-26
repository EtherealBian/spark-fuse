"""The live CLI and offline CLI must have byte-identical stdout, with progress on stderr."""
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import tile235


class Channels(unittest.TestCase):
    def test_cli_run_then_replay(self):
        ref = Path('reports/qwen235b')
        rows = json.loads((ref / 'rows.json').read_bytes())
        def measured(port, save):
            print('case progress', flush=True)
            save(rows)
            return rows
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            (d / 'runtime.json').write_bytes((ref / 'runtime.json').read_bytes())
            def call(mode):
                out, err = StringIO(), StringIO()
                with patch('sys.argv', ['tile235', mode, '--block', '0', '--out', t, '--port', '1']), redirect_stdout(out), redirect_stderr(err):
                    tile235.main()
                return out.getvalue(), err.getvalue()
            with patch.object(tile235, 'run', measured):
                live, progress = call('run')
            replay, quiet = call('audit')
            self.assertEqual((live, progress, quiet), (replay, 'case progress\n', ''))
            self.assertEqual(json.loads(live)['output_per_request'], 16)
