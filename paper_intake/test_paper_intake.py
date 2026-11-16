import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from reportlab.pdfgen import canvas
from paper_intake import (github_url, git, inspect_repo, extract_pdf, analyze_paper,
                          page_chunks, main)

class IntakeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pdf = self.root / 'paper.pdf'
        doc = canvas.Canvas(str(self.pdf))
        doc.drawString(50, 750, 'We use CIFAR-10 and train the model for 100 epochs.')
        doc.showPage()
        doc.save()
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        git(self.repo, 'init')
        git(self.repo, 'remote', 'add', 'origin', 'https://github.com/example/research.git')
        (self.repo / 'README.md').write_text('Train with python train.py', encoding='utf-8')
        (self.repo / 'train.py').write_text("raise RuntimeError('MUST NOT RUN')\ndef train():\n    pass\n", encoding='utf-8')
        (self.repo / '.env').write_text('SECRET=do-not-read', encoding='utf-8')
        git(self.repo, 'add', '.')
        git(self.repo, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-m', 'fixture')
        self.page = dict(page=1, text='We use CIFAR-10 and train the model for 100 epochs.', tables=[])

    def response(self, quote='We use CIFAR-10', page=1):
        data = dict(findings=[dict(category='dataset', statement='The dataset is CIFAR-10.',
                                  evidence=[dict(page=page, quote=quote)])], unknowns=['Seed not specified in excerpt'])
        return dict(status='completed', output=[dict(type='message', content=[dict(type='output_text', text=json.dumps(data))])])

    def test_url_validation(self):
        self.assertEqual(github_url('https://github.com/a/b.git/'), 'https://github.com/a/b.git')
        for url in ('file:///tmp/repo', 'https://evil.test/a/b', 'https://github.com/a/b/tree/main', '--upload-pack=evil'):
            with self.assertRaises(ValueError):
                github_url(url)

    def test_pdf_text_and_render(self):
        output = self.root / 'extracted'
        output.mkdir()
        paper = extract_pdf(self.pdf, output, render=True)
        self.assertIn('CIFAR-10', paper['pages'][0]['text'])
        self.assertTrue((output / 'pages/page-0001.png').is_file())
        self.assertEqual(len(paper['sha256']), 64)

    def test_blank_pdf_warns(self):
        blank = self.root / 'blank.pdf'
        doc = canvas.Canvas(str(blank)); doc.showPage(); doc.save()
        paper = extract_pdf(blank, self.root)
        self.assertTrue(paper['pages'][0]['warnings'])

    def test_repo_snapshot_never_runs_code(self):
        (self.repo / 'README.md').write_text('Uncommitted content', encoding='utf-8')
        result = inspect_repo('https://github.com/example/research', self.root, repo_path=self.repo)
        docs = {d['path']: d for d in result['documents']}
        self.assertEqual(docs['README.md']['content'], 'Train with python train.py')
        self.assertNotIn('.env', docs)
        self.assertEqual(docs['train.py']['symbols'][0]['name'], 'train')
        self.assertEqual(len(result['commit']), 40)

    def test_origin_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'origin'):
            inspect_repo('https://github.com/other/repo', self.root, repo_path=self.repo)

    def test_size_limits_report_skips(self):
        result = inspect_repo('https://github.com/example/research', self.root, repo_path=self.repo, max_file_bytes=10)
        self.assertEqual(result['documents'], [])
        self.assertTrue(any(s['reason'] == 'file size limit' for s in result['skipped']))

    def test_model_analysis_with_evidence(self):
        payloads = []
        def transport(payload):
            payloads.append(payload)
            return self.response()
        result = analyze_paper({'pages': [self.page]}, 'Identify the dataset', 'test-model', transport=transport)
        self.assertEqual(result['findings'][0]['evidence'][0]['page'], 1)
        self.assertTrue(payloads[0]['text']['format']['strict'])
        self.assertFalse(payloads[0]['store'])

    def test_fabricated_evidence_rejected(self):
        for quote, page in [('Invented unsupported quote', 1), ('We use CIFAR-10', 9)]:
            with self.assertRaisesRegex(ValueError, 'Unverifiable'):
                analyze_paper({'pages': [self.page]}, 'test', 'test-model', transport=lambda _: self.response(quote, page))

    def test_refusal_and_incomplete_rejected(self):
        responses = [dict(status='incomplete'), dict(status='completed', output=[dict(type='message', content=[dict(type='refusal')])])]
        for response in responses:
            with self.assertRaises(ValueError):
                analyze_paper({'pages': [self.page]}, 'test', 'test-model', transport=lambda _: response)

    def test_api_key_required(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'OPENAI_API_KEY'):
                analyze_paper({'pages': [self.page]}, 'test', 'test-model')

    def test_chunking_does_not_drop_pages(self):
        pages = [dict(self.page, page=i) for i in range(1, 6)]
        chunks = list(page_chunks(pages, limit=200))
        self.assertEqual([p['page'] for c in chunks for p in c], list(range(1, 6)))
        with self.assertRaises(ValueError):
            list(page_chunks(pages, limit=10))

    def test_cli_end_to_end_and_existing_output(self):
        output = self.root / 'result'
        args = ['--paper', str(self.pdf), '--github', 'https://github.com/example/research',
                '--repo-path', str(self.repo), '--objective', 'Inspect training', '--output', str(output)]
        self.assertEqual(main(args), 0)
        self.assertEqual(json.loads((output / 'manifest.json').read_text())['status'], 'complete')
        self.assertTrue((output / 'repository.json').is_file())
        self.assertIn('Repository entry-point hints', (output / 'report.md').read_text())
        self.assertFalse((output / 'analysis.json').exists())
        with self.assertRaises(SystemExit):
            main(args)

    def test_cli_failure_preserves_extraction(self):
        output = self.root / 'failed'
        args = ['--paper', str(self.pdf), '--github', 'https://github.com/wrong/repo',
                '--repo-path', str(self.repo), '--objective', 'Inspect', '--output', str(output)]
        self.assertEqual(main(args), 1)
        self.assertEqual(json.loads((output / 'manifest.json').read_text())['status'], 'error')
        self.assertTrue((output / 'paper.json').is_file())

    def test_unreadable_pages_block_analysis(self):
        with self.assertRaisesRegex(ValueError, 'Unreadable'):
            analyze_paper({'pages': [dict(self.page, text='')]}, 'test', 'test-model', transport=lambda _: self.response())

    def test_invalid_model_schema_rejected(self):
        response = dict(status='completed', output=[dict(type='message', content=[dict(type='output_text', text='{"findings": []}')])])
        with self.assertRaises(ValueError):
            analyze_paper({'pages': [self.page]}, 'test', 'test-model', transport=lambda _: response)

    def test_analysis_preserves_every_chunk(self):
        pages = [dict(self.page, page=i) for i in range(1, 4)]
        def transport(payload):
            page = json.loads(payload['input'])['pages'][0]['page']
            return self.response(page=page)
        result = analyze_paper({'pages': pages}, 'test', 'test-model', chunk_chars=150, transport=transport)
        self.assertEqual([f['evidence'][0]['page'] for f in result['findings']], [1, 2, 3])

    def test_sparse_page_ocr(self):
        from unittest.mock import MagicMock
        import sys
        blank = self.root / 'scan.pdf'
        doc = canvas.Canvas(str(blank)); doc.showPage(); doc.save()
        fake = MagicMock()
        fake.image_to_string.return_value = self.page['text']
        with patch.dict(sys.modules, {'pytesseract': fake}):
            result = extract_pdf(blank, self.root, ocr=True)
        self.assertEqual(result['pages'][0]['extraction'], 'ocr')
        self.assertIn('CIFAR-10', result['pages'][0]['text'])

if __name__ == '__main__':
    unittest.main()
