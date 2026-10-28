import re,subprocess
rd=lambda p:open(f'parts/{p}.tex').read()
INL=lambda t:re.sub(r'\\input\{(tab\d*_\w+\.tex)\}',lambda m:open(m.group(1)).read().strip(),t)
m1,m2,m3,p1,p2,p3,pmlp,p3b,bib=[rd(x) for x in ['m1','m2','m3','p1','p2','p3','pmlp','p3b','bib']]
# ---------- bibliography
bhead,bitems=bib.split('\\begin{thebibliography}')[1].split('\n',1)[0],re.split(r'(?=\\bibitem\{)',bib.split('\\begin{thebibliography}')[1].split('\n',1)[1].replace('\\end{thebibliography}','').replace('\\end{document}',''))
bitems=[b for b in bitems if b.strip().startswith('\\bibitem')]
def cited(t): return {k.strip() for m in re.finditer(r'\\cite\{([^}]*)\}',t) for k in m.group(1).split(',')}
def bibfor(t):
    c=cited(t); items=[b for b in bitems if re.match(r'\\bibitem\{([^}]*)\}',b.strip()).group(1) in c]
    return "\\begin{thebibliography}{99}\\small\n"+"".join(items)+"\n\\end{thebibliography}\n"
# ---------- supplement text
a=p3.index('\\appendices'); p3main=p3[:a]
d=p3main.index('\\section{Discussion and Limitations}')
rob_res=p3main[:d]; disc=p3main[d:]; disc=disc[:disc.index('\\section{Conclusion}')]
k=p3b.index('\\section*{Reproducibility}'); repro_long=p3b[k:]; supp_rest=p3b[:k]
meth=p1[p1.index('\\section{Methods and Experimental Setup}'):]
meth=meth.replace('\\section{Methods and Experimental Setup}','\\section{Extended methods and experimental setup}')
res=p2.replace('\\section{Results}','\\section{Extended results}')
rob_res=rob_res.replace('\\subsection{Robustness checks}','\\subsection{Robustness checks}')
disc=disc.replace('\\section{Discussion and Limitations}','\\section{Extended discussion and limitations}')
supp_body=INL(meth+"\n"+res+"\n"+rob_res+"\n"+disc+"\n"+supp_rest+"\n"+pmlp+"\n")
tabmap={}; n=0
for m in re.finditer(r'\\begin\{table\*?\}|\\label\{(tab:[a-z0-9]+)\}',supp_body):
    if m.group(0).startswith('\\begin'): n+=1
    else: tabmap[m.group(1)]='S%d'%n
secmap={}; L=0
for m in re.finditer(r'\\section\{[^}]*\}|\\label\{((?:sec|app):[a-z0-9]+)\}',supp_body):
    if m.group(0).startswith('\\section'): L+=1
    elif m.group(1): secmap[m.group(1)]=chr(64+L)
print(tabmap); print(secmap)
def sref(t):
    def f(m):
        l=m.group(1)
        if l in tabmap: return 'Table~'+tabmap[l]
        if l in secmap: return 'Supplement Sec.~'+secmap[l]
        raise SystemExit('unresolved sref '+l)
    return re.sub(r'\\sref\{([^}]*)\}',f,t)
repro_main=r"""\section*{Reproducibility}
Code, cleaned BRFSS/UCI tables, per-seed results for every table and a script that checks the numbers quoted in the text are at \texttt{github.com/pradhuman001/fedcvd-fairness}. The supplementary material (Sections A--H, Tables S1--S%d: extended methods and results, robustness, MLP, convergence, exponentiated-gradient reference, UCI replication) accompanies the paper and is in the repository (\texttt{supplement.pdf}).

""" % n
main_src=sref(INL(m1+"\n"+m2+"\n"+m3+"\n"+repro_main))
main_src=main_src+bibfor(main_src)+"\\end{document}\n"
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
This supplement contains the extended methods, results and discussion of the main paper, additional robustness checks, an MLP, convergence checks, an exponentiated-gradient reference and a UCI replication. Table numbers (S1, S2, \\dots) and section letters match the references in the main paper. Section references of the form ``Sec.~III-A'' inside copied text refer to Section~A here.

"""+supp_body+"\n"+repro_long+"\n"
supp=supp+bibfor(supp)+"\\end{document}\n"
open('main.tex','w').write(main_src); open('supplement.tex','w').write(supp)
def build(name,src):
    have=subprocess.run(['kpsewhich','IEEEtran.cls'],capture_output=True,text=True).stdout.strip()
    t=src if have else src.replace(r"\documentclass[conference]{IEEEtran}",FB)
    open(name+'_build.tex','w').write(t)
    for _ in range(2): r=subprocess.run(['pdflatex','-interaction=nonstopmode',name+'_build.tex'],capture_output=True,text=True)
    print(name,[l for l in r.stdout.splitlines() if l.startswith('!') or 'undefined' in l or 'Overfull \\hbox (' in l][:10])
build('paper',main_src); build('supplement',supp)
