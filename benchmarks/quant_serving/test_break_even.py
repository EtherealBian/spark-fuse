import copy,json,tempfile,unittest
from pathlib import Path
from audit_break_even import compare,crossing,rebuild
from test_heldout import fixture

class TestBreakEven(unittest.TestCase):
    def test_crossing_cannot_hide_later_reversal(self):
        self.assertIsNone(crossing([True,False,True,False]));self.assertEqual(crossing([True,False,True,True]),3)
        for xs in ([],[1]*4):
            with self.assertRaises(ValueError):crossing(xs)

    def test_preparation_cost_and_repeat_quality(self):
        z=fixture()[0]
        for a in z['runs']:
            a['cumulative_wall_ms']={'cold':[5,10,15,20],'prepared':[7,9,12,15],'on-demand':[5,8,10,12]}[a['arm']].copy()
        self.assertEqual(compare(z,'cold','prepared')['output_preserving_first_sustained_request'],2);self.assertIsNone(compare(z,'on-demand','prepared')['timing_only_first_sustained_request'])
        next(a for a in z['runs'] if a['arm']=='prepared' and a['repeat']==1)['cumulative_wall_ms'][-1]=22
        self.assertIsNone(compare(z,'cold','prepared')['timing_only_first_sustained_request'])
        for a in z['runs']:
            if a['arm']=='prepared':a['cumulative_wall_ms'][-1]=15
        next(a for a in z['runs'] if a['arm']=='prepared')['requests'][0]['ids'][0]=99
        v=compare(z,'cold','prepared');self.assertEqual(v['timing_only_first_sustained_request'],2);self.assertIsNone(v['output_preserving_first_sustained_request'])

    def test_rebuild_binds_real_published_runtime_and_negative_inputs(self):
        ref=Path('reports/qwen72b');r=fixture()
        with tempfile.TemporaryDirectory() as td:
            d=Path(td)
            for n in ('runtime','checkpoint'):(d/(n+'.json')).write_bytes((ref/('prefix-'+n+'.json')).read_bytes())
            (d/'rows.json').write_text(json.dumps(r));out=rebuild(d,ref);self.assertEqual((out['target_requests'],out['actual_preparation_requests']),(144,12));self.assertEqual(len(out['cases']),4);self.assertEqual(len(out['evidence_sha256']),6)
            rt=json.loads((d/'runtime.json').read_bytes());rt['server_sha256']='0'*64;(d/'runtime.json').write_text(json.dumps(rt))
            with self.assertRaises(ValueError):rebuild(d,ref)

if __name__=='__main__':unittest.main()
