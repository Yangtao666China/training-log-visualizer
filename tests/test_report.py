import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from trainlens import load_run, render_report, summarize


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def run_file(self, text, name="run.csv"):
        path = self.root/name
        path.write_text(text,encoding="utf-8")
        return load_run(path)

    def test_best_uses_validation_and_first_tie(self):
        run = self.run_file("epoch,train_loss,val_loss\n1,.4,.5\n2,.3,.4\n3,.2,.4\n")
        summary = summarize(run)
        self.assertEqual(summary["best_epoch"],2)
        self.assertAlmostEqual(summary["gap_at_best"],.1)

    def test_accuracy_max(self):
        run = self.run_file("epoch,train_loss,val_loss,val_acc\n1,.3,.4,.7\n2,.2,.3,.8\n")
        self.assertEqual(summarize(run,"val_acc","max")["best_epoch"],2)

    def test_reject_nonfinite(self):
        for bad in ["nan","inf","-inf"]:
            with self.subTest(bad=bad),self.assertRaisesRegex(ValueError,"line 2"):
                self.run_file(f"epoch,train_loss,val_loss\n1,.4,{bad}\n")

    def test_reject_invalid_epoch_order(self):
        for epoch in ["1","0","1.5","-1"]:
            with self.subTest(epoch=epoch),self.assertRaises(ValueError):
                self.run_file(f"epoch,train_loss,val_loss\n1,.4,.5\n{epoch},.3,.4\n")

    def test_reject_bad_headers(self):
        for text in ["", "epoch,train_loss\n1,.4\n", "epoch,train_loss,val_loss,val_loss\n1,.4,.5,.5\n"]:
            with self.subTest(text=text),self.assertRaises(ValueError):
                self.run_file(text)

    def test_reject_missing_or_extra_values(self):
        for row in ["1,.4,", "1,.4,.5,extra", "1,.4"]:
            with self.subTest(row=row),self.assertRaises(ValueError):
                self.run_file("epoch,train_loss,val_loss\n"+row+"\n")

    def test_accuracy_range(self):
        with self.assertRaises(ValueError):
            self.run_file("epoch,train_loss,val_loss,val_acc\n1,.3,.4,90\n")

    def test_utf8_bom_and_extra_columns(self):
        run = self.run_file("\ufeffepoch,train_loss,val_loss,notes\n1,.3,.4,hello\n")
        self.assertEqual(len(run["rows"]),1)

    def test_report_escapes_html_and_script_data(self):
        run = self.run_file("epoch,train_loss,val_loss\n1,.3,.4\n")
        run["name"] = '</script><script>alert("x")</script>'
        target = self.root/"report.html"
        render_report([run],target,title="<img src=x onerror=alert(1)>")
        text = target.read_text(encoding="utf-8")
        self.assertNotIn(run["name"],text)
        self.assertNotIn("<img src=x",text)
        self.assertIn("\\u003c/script>",text)
        self.assertEqual(json.loads(target.with_suffix(".json").read_text())[0]["name"],run["name"])

    def test_duplicate_run_names(self):
        run = self.run_file("epoch,train_loss,val_loss\n1,.3,.4\n")
        with self.assertRaises(ValueError):
            render_report([run,run],self.root/"report.html")

    def test_missing_selection_metric(self):
        run = self.run_file("epoch,train_loss,val_loss\n1,.3,.4\n")
        with self.assertRaises(ValueError):
            summarize(run,"val_acc","max")

    def test_cli_pattern_and_errors(self):
        self.run_file("epoch,train_loss,val_loss\n1,.3,.4\n")
        result = subprocess.run([sys.executable,"-m","trainlens",str(self.root/"*.csv"),
                                 "--output",str(self.root/"report.html")],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertTrue((self.root/"report.html").exists())
        bad = subprocess.run([sys.executable,"-m","trainlens",str(self.root/"missing.csv")],capture_output=True,text=True)
        self.assertEqual(bad.returncode,2)


if __name__ == "__main__":
    unittest.main()
