import re,subprocess
rd=lambda p:open(f'parts/{p}.tex').read()
INL=lambda t:re.sub(r'\\input\{(tab\d*_\w+\.tex)\}',lambda m:open(m.group(1)).read().strip(),t)
p1,p2,p3,pmlp,p3b,bib=[rd(x) for x in ['p1','p2','p3','pmlp','p3b','bib']]
a=p3.index('\\appendices'); main_body=p1+"\n"+p2+"\n"+p3[:a]
# p3b: robust, conv, eg, uci, reproducibility
k=p3b.index('\\section*{Reproducibility}'); repro=p3b[k:]; supp_rest=p3b[:k]
r0=supp_rest.index('\\section{Convergence of penalized models}')
robust=supp_rest[:r0]; rest=supp_rest[r0:]
APP=['tab:grp','tab:perlam','tab:ctl','tab:lso','tab:rob','tab:mlp','tab:mlpoff','tab:fine','tab:glob']
moved=[]
for lab in APP:
    for name in ('main_body','pmlp'):
        src=globals()[name]
        if '\\label{'+lab+'}' not in src: continue
        i=src.index('\\label{'+lab+'}'); a2=src.rfind('\\begin{table',0,i); e=src.index('\\end{table',i); e=src.index('}',e)+1
        if src[e:e+1]=='*': e+=1
        moved.append(src[a2:e]); globals()[name]=src[:a2]+src[a2+len(src[a2:e]):]; break
    else: raise SystemExit('missing '+lab)
# supplement source, in order
addtab='\\section{Additional tables}\\label{app:tables}\nExploratory and per-$\\lambda$ tables referred to in the main text.\n\n'+'\n\n'.join(moved)+'\n\n'
supp_body=addtab+robust+'\n'+pmlp+'\n'+rest
supp_body=INL(supp_body)
# label -> number / section letter
tabmap={}; n=0
for m in re.finditer(r'\\begin\{table\*?\}|\\label\{(tab:[a-z0-9]+)\}',supp_body):
    if m.group(0).startswith('\\begin'): n+=1
    else: tabmap[m.group(1)]='S%d'%n
secmap={}; L=0
for m in re.finditer(r'\\section\{[^}]*\}(?:\s*\\label\{([a-z]+:[a-z0-9]+)\})?|\\label\{(sec:[a-z0-9]+|app:[a-z0-9]+)\}',supp_body):
    if m.group(0).startswith('\\section'): L+=1
    lab=m.group(1) or m.group(2)
    if lab: secmap[lab]=chr(64+L)
print(tabmap,secmap)
def fix(t):
    t=re.sub(r'(?:Appendix|Sec\.)~\\ref\{((?:app|sec):[a-z0-9]+)\}',lambda m:('Supplement Sec.~'+secmap[m.group(1)]) if m.group(1) in secmap else m.group(0),t)
    t=re.sub(r'\\ref\{(tab:[a-z0-9]+)\}',lambda m:tabmap[m.group(1)] if m.group(1) in tabmap else m.group(0),t)
    return t
main_body=fix(main_body)
main_src=INL(main_body+"\n"+repro+"\n"+"\\begin{thebibliography}"+bib.split('\\begin{thebibliography}')[1].replace('\\end{document}','')+"\n\\end{document}\n")
# repro mentions Table~\ref{tab:seeds} fine (main)
FB=r"""\documentclass[10pt,twocolumn]{article}
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
head=p1[:p1.index('\\begin{document}')]
supp=head+"""\\begin{document}
\\title{Supplementary Material: What Do Fairness Mitigations Do to Intersectional Gaps in Cardiovascular Risk Prediction?}
\\author{\\IEEEauthorblockN{Pradhuman Kumar Sharma}
\\IEEEauthorblockA{Department of Computer Science and Engineering, Sharda University, Greater Noida, India}}
\\maketitle
\\renewcommand{\\thetable}{S\\arabic{table}}
\\appendices
This supplement contains the additional tables, robustness checks, MLP results, convergence checks, the exponentiated-gradient reference and the UCI replication referred to in the main paper. Section letters and table numbers (S1, S2, \\dots) match the references in the main text.

"""+supp_body+"\n"+"\\begin{thebibliography}{9}\\small\n"+re.search(r"\\bibitem\{agarwal2018\}.*",bib).group(0)+"\n\\end{thebibliography}\n\\end{document}\n"
open('main.tex','w').write(main_src); open('supplement.tex','w').write(supp)
def build(name,src):
    have=subprocess.run(['kpsewhich','IEEEtran.cls'],capture_output=True,text=True).stdout.strip()
    t=src if have else src.replace(r"\documentclass[conference]{IEEEtran}",FB)
    open(name+'_build.tex','w').write(t)
    for _ in range(2): r=subprocess.run(['pdflatex','-interaction=nonstopmode',name+'_build.tex'],capture_output=True,text=True)
    print(name,[l for l in r.stdout.splitlines() if l.startswith('!') or 'undefined' in l or 'Overfull \\hbox (' in l][:10])
build('paper',main_src); build('supplement',supp)
