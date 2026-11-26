from unittest import TestCase
from wave_ledger import intervals


class Intervals(TestCase):
    def row(self):
        return dict(arrival_ns=1,blocker={'release_start_ns':8},grant_ns=10,targets=[{'end_ns':17},{'end_ns':15}])

    def test_later_completion_not_sum_of_overlapping_responses(self):
        r=self.row(); expected=dict(blocker=7,handoff=2,after_grant=7,total=16)
        self.assertEqual(intervals(r),expected);r['targets'].reverse();self.assertEqual(intervals(r),expected)

    def test_equal_release_and_grant_is_valid(self):
        r=self.row();r['grant_ns']=8;self.assertEqual(intervals(r)['handoff'],0)

    def test_reverse_floating_and_missing_boundaries_rejected(self):
        for key,v in [('arrival_ns',10),('grant_ns',7),('grant_ns',10.0),('targets',[{'end_ns':17}]),('targets',[{'end_ns':9},{'end_ns':20}])]:
            with self.subTest(key=key,v=v),self.assertRaises(ValueError):
                r=self.row();r[key]=v;intervals(r)
