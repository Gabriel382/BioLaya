#!/usr/bin/env python3
"""Push only a VALIDATED native Ollama model; requires user opt-in."""
import argparse,subprocess

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--local',default='biolaya-systemone:421m')
 p.add_argument('--remote',required=True,help='e.g. Gabriel382/biolaya-systemone:421m')
 p.add_argument('--confirm',action='store_true');a=p.parse_args()
 print('Source:',a.local,'Destination:',a.remote)
 print('Requirement: scripts/test_native_ollama.py and compare_native_parity.py passed')
 if not a.confirm:
    print('Dry-run only. Add --confirm to copy and push.');return
 subprocess.run(['ollama','show',a.local],check=True)
 subprocess.run(['ollama','cp',a.local,a.remote],check=True)
 subprocess.run(['ollama','push',a.remote],check=True)
 print('Pushed:',a.remote)
if __name__=='__main__':main()
