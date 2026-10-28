import re,subprocess,os,shutil
parts=['p1','p2','p3','pmlp','p3b','bib']
s="".join(open(f'parts/{p}.tex').read()+"\n" for p in parts)
APP=['tab:grp','tab:perlam','tab:ctl','tab:lso','tab:rob','tab:mlp','tab:mlpoff']
moved=[]
for lab in APP:
    i=s.index('\\label{'+lab+'}'); a=s.rfind('\\begin{table',0,i); e=s.index('\\end{table',i); e=s.index('}',e)+1
    if s[e:e+1]=='*': e+=1
    blk=s[a:e]
    if not blk.rstrip().endswith('}'): blk=s[a:s.index('\n',e)]
    moved.append(blk); s=s[:a]+s[a+len(blk):]
k=s.index('\\section*{Reproducibility}')
s=s[:k]+'\\section{Additional tables}\\label{app:tables}\nExploratory and per-$\\lambda$ tables referred to in the text.\n\n'+'\n\n'.join(moved)+'\n\n'+s[k:]
INL=lambda t:re.sub(r'\\input\{(tab\d*_\w+\.tex)\}',lambda m:open(m.group(1)).read().strip(),t)
s=INL(s)
open('main.tex','w').write(s)
have=subprocess.run(['kpsewhich','IEEEtran.cls'],capture_output=True,text=True).stdout.strip()
if not have:  # IEEE-like fallback preview (real IEEEtran is used on Overleaf)
    fb=r"""\documentclass[10pt,twocolumn]{article}
\usepackage[letterpaper,top=0.75in,bottom=1in,left=0.625in,right=0.625in,columnsep=0.25in]{geometry}
\usepackage{times}
\renewcommand\thesection{\Roman{section}}
\renewcommand\thesubsection{\thesection-\Alph{subsection}}
\newcommand{\IEEEauthorblockN}[1]{\textbf{#1}\\}
\newcommand{\IEEEauthorblockA}[1]{\small #1}
\newenvironment{IEEEkeywords}{\noindent\textbf{\textit{Index Terms}}---}{\par}
\newcommand{\appendices}{\setcounter{section}{0}\renewcommand\thesection{\Alph{section}}\renewcommand\thesubsection{\thesection.\arabic{subsection}}}
\date{}\setlength{\parindent}{1em}\setlength{\parskip}{0pt}
"""
    s=s.replace(r"\documentclass[conference]{IEEEtran}",fb)
    s=s.replace(r"\section*{Reproducibility}",r"\section*{Reproducibility}")
open('paper_build.tex','w').write(s)
for _ in range(2):
    r=subprocess.run(['pdflatex','-interaction=nonstopmode','paper_build.tex'],capture_output=True,text=True)
print([l for l in r.stdout.splitlines() if l.startswith('!') or 'undefined' in l or 'Overfull \\hbox (' in l][:15])
