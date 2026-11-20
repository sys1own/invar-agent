"""Executable provenance reports and standalone receipt verification."""
from .base import package_name, selected


REPORT = '''from html import escape
from pathlib import Path
from ..sdk import MODEL,simulate

def generate(output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    run=simulate()
    lines=['# Reproducibility report',MODEL['ir']['title'],MODEL['ir']['summary'],'## Equations']
    for equation in MODEL['ir']['equations']:
        source=equation['provenance']
        lines.append(equation['canonical_latex']+' — '+source['file']+': '+source['location'])
    lines.append('## Constants (SI)')
    for name,value in run.parameters.items():
        lines.append(name+' = '+str(value)+' '+MODEL['parameter_definitions'][name]['dimension']['canonical_unit'])
    lines.append('## Reproduced final state and invariant contracts')
    for name,values in run.trajectories.items(): lines.append(name+' = '+str(values[-1]))
    lines.append(str(run.invariant_status))
    text='\\n\\n'.join(lines)+'\\n'
    (output/'reproducibility_report.md').write_text(text,encoding='utf-8')
    (output/'reproducibility_report.html').write_text('<!doctype html><meta charset="utf-8"><h1>Reproducibility report</h1><pre>'+escape(text)+'</pre>',encoding='utf-8')
    rows=['\\\\begin{tabular}{lr}','Symbol & SI value \\\\\\\\']
    for name,value in run.parameters.items(): rows.append(name.replace('_','\\\\_')+' & '+str(value)+' \\\\\\\\')
    rows.append('\\\\end{tabular}')
    (output/'reproducibility_table.tex').write_text('\\n'.join(rows)+'\\n',encoding='utf-8')
    return [str(output/name) for name in ('reproducibility_report.md','reproducibility_report.html','reproducibility_table.tex')]
'''

CLI = '''import argparse
from .report_generator import generate

def main():
    parser=argparse.ArgumentParser(description='Reproduce the verified research instrument')
    parser.add_argument('command',choices=['run'])
    parser.add_argument('--output',default='audit_out')
    args=parser.parse_args()
    for path in generate(args.output): print(path)

if __name__=='__main__':
    main()
'''

VERIFY = '''import argparse
import json
from pathlib import Path
import blake3

def verify(root=None):
    if root is None and (Path.cwd()/'provenance'/'build_receipt.json').is_file():
        root=Path.cwd()
    if root is None:
        root=next((path for path in Path(__file__).resolve().parents if (path/'provenance'/'build_receipt.json').is_file()),None)
    if root is None:
        raise ValueError('Build receipt not found; specify --root')
    root=Path(root).resolve()
    receipt=json.loads((root/'provenance'/'build_receipt.json').read_text())
    expected=receipt.pop('merkle_root_blake3')
    payload=json.dumps(receipt,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode('utf-8')
    actual=blake3.blake3(payload,derive_key_context='invar.receipt.merkle.v1').hexdigest()
    failures=[]
    if expected!=actual: failures.append('build_receipt.json: Merkle root mismatch')
    for artifact in receipt['artifacts']:
        path=(root/artifact['relpath']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            failures.append(artifact['relpath']+': missing or unsafe path')
        elif blake3.blake3(path.read_bytes()).hexdigest()!=artifact['content_blake3']:
            failures.append(artifact['relpath']+': content BLAKE3 mismatch')
    if failures: raise ValueError('\\n'.join(failures))
    return True

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root')
    args=parser.parse_args()
    try:
        verify(args.root)
    except (ValueError,OSError,KeyError) as error:
        parser.exit(1,str(error)+'\\n')
    print('Receipt verified: every artifact matches BLAKE3')

if __name__=='__main__': main()
'''


def emit(node,context):
    prefix=f"src/{package_name(context)}/audit"
    return selected(node,{f"{prefix}/cli.py":CLI,f"{prefix}/report_generator.py":REPORT,f"{prefix}/verify_receipt.py":VERIFY})
