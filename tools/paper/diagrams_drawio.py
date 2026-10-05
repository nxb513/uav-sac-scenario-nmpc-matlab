"""Write the two algorithm diagrams as draw.io files (rendered to PDF by the draw.io desktop app).

Figure_1.drawio : the three offline stages (SAC tunes the teacher, DAgger distills the linear residual,
                      confidences), as implemented in experiments/run_d1_joint_pipeline.m and src/joint.
Figure_2.drawio: the deployed controller P (d1_student_feature, d1_blend_control).

Usage: python diagrams_drawio.py <out_dir>; then export each file with the draw.io desktop app,
for example: draw.io --export --format pdf --crop -o Figure_1.pdf Figure_1.drawio
"""
import os
import sys
from xml.sax.saxutils import escape

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else '.'

FONT = 'fontFamily=Times New Roman;fontSize=12;fontColor=#0b0b0b;'
BOX = 'rounded=1;whiteSpace=wrap;html=1;arcSize=8;strokeColor=#52514e;' + FONT
GROUP = ('rounded=1;whiteSpace=wrap;html=1;arcSize=3;fillColor=#f4f3f0;strokeColor=#9a9893;dashed=0;'
         'verticalAlign=top;align=left;spacingLeft=8;spacingTop=4;fontStyle=1;' + FONT)
FILL = {'teacher': 'fillColor=#fde3d8;', 'student': 'fillColor=#d9e8fa;', 'lqr': 'fillColor=#d6f2e7;',
        'plain': 'fillColor=#ffffff;', 'out': 'fillColor=#ffffff;strokeWidth=2;'}
EDGE = ('edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;strokeColor=#52514e;'
        + FONT.replace('fontSize=12', 'fontSize=11'))


class Diagram:
    def __init__(self, name):
        self.name, self.cells, self.n = name, [], 2

    def _id(self):
        self.n += 1
        return 'c%d' % self.n

    def box(self, x, y, w, h, html, kind='plain', group=False):
        i = self._id()
        style = GROUP if group else BOX + FILL[kind]
        self.cells.append('<mxCell id="%s" value="%s" style="%s" vertex="1" parent="1"><mxGeometry x="%d" y="%d" '
                          'width="%d" height="%d" as="geometry"/></mxCell>' % (i, escape(html, {'"': '&quot;'}), style, x, y, w, h))
        return i

    def edge(self, a, b, label='', extra='', points=()):
        i = self._id()
        pts = ''.join('<mxPoint x="%d" y="%d"/>' % p for p in points)
        geo = '<mxGeometry relative="1" as="geometry">%s</mxGeometry>' % (('<Array as="points">%s</Array>' % pts) if pts else '')
        self.cells.append('<mxCell id="%s" value="%s" style="%s" edge="1" parent="1" source="%s" target="%s">%s</mxCell>'
                          % (i, escape(label, {'"': '&quot;'}), EDGE + extra, a, b, geo))
        return i

    def write(self, path):
        body = ''.join(self.cells)
        xml = ('<mxfile><diagram name="%s"><mxGraphModel><root><mxCell id="0"/><mxCell id="1" parent="0"/>%s'
               '</root></mxGraphModel></diagram></mxfile>' % (self.name, body))
        open(path, 'w', encoding='utf-8').write(xml)


FH = 'F&#770;'   # F with combining circumflex

# ---------------- Fig. training pipeline ----------------
d = Diagram('training')
d.box(10, 10, 330, 470, '1&nbsp; SAC tunes the teacher weights<br><span style="font-weight:normal">(iterations 1&ndash;100, one-step bandit)</span>', group=True)
a1 = d.box(30, 70, 290, 48, 'SAC policy: Gaussian, unbounded;<br>action <i>a</i> &isin; &#8477;<sup>6</sup>')
a2 = d.box(30, 138, 290, 48, '<i>Q</i>, <i>R</i> = base groups &times; 10<sup><i>a</i></sup><br>(base: Bryson rule, or random)')
a3 = d.box(30, 206, 290, 70, '<b>Scenario NMPC teacher</b><br><i>M</i> = 5 model scenarios, <i>N</i> = 20, <i>N</i><sub>c</sub> = 5<br>'
           'receives the current wind force <i>F<sub>k</sub></i> (privileged)', 'teacher')
a4 = d.box(30, 296, 290, 62, '20 flights per iteration: nominal plant,<br>synthetic wind (mean 1&ndash;10 m/s,<br>Dryden gusts, rotor drag)')
a5 = d.box(30, 378, 290, 62, 'reward <i>r</i>(<i>a</i>) = &minus;(RMSE<sub>pos</sub> + 0.01 RMS&Delta;<i>u</i><br>+ 0.5 <i>r</i><sub>viol</sub> + 5 <i>r</i><sub>fail</sub>)'
           '<br>stored in the replay buffer; SAC update')
d.edge(a1, a2); d.edge(a2, a3); d.edge(a3, a4); d.edge(a4, a5)
d.edge(a5, a1, extra='exitX=0;exitY=0.5;entryX=0;entryY=0.5;', points=[(20, 409), (20, 94)])

d.box(360, 10, 400, 470, '2&nbsp; DAgger distills a linear residual<br><span style="font-weight:normal">(teacher frozen at the chosen SAC iteration, <i>a</i> = &mu;)</span>', group=True)
b1 = d.box(380, 70, 175, 62, 'DAgger iteration 1:<br>teacher flies 20 flights', 'teacher')
b2 = d.box(565, 70, 175, 62, 'iterations 2&ndash;10: student flies<br>(&alpha; = 1); teacher only<br>queried at visited states', 'student')
b3 = d.box(380, 152, 360, 62, 'label at usable, in-envelope states:<br><i>y</i> = (<i>u</i><sub>T</sub> &minus; sat(<i>u</i><sub>LQR</sub>)) &#8856; '
           '(<i>u</i><sub>max</sub> &minus; <i>u</i><sub>min</sub>), with features &phi; (27, incl. ' + FH + ')')
b4 = d.box(380, 234, 360, 48, 'aggregate per-flight sufficient statistics;<br>ridge regression, &lambda; by 5-fold cross-validation over flights')
b5 = d.box(380, 302, 360, 48, 'linearized check at hover: &rho;(<i>A</i><sub>&alpha;</sub>) &lt; 1 for every &alpha; &isin; [0, 1]')
b6 = d.box(380, 370, 360, 62, '<i>W</i>* = candidate with the lowest validation RMSE<br>(15 held-out-wind flights) among the stable ones', 'student')
d.edge(b1, b3); d.edge(b2, b3); d.edge(b3, b4); d.edge(b4, b5); d.edge(b5, b6)
d.edge(b4, b2, label='next student', extra='dashed=1;exitX=1;exitY=0.5;entryX=1;entryY=0.5;', points=[(752, 258), (752, 101)])
d.edge(a3, b1, label='teacher at the<br>chosen iteration', extra='exitX=1;exitY=0.5;entryX=0;entryY=0.5;', points=[(350, 241), (350, 101)])

d.box(780, 10, 300, 300, '3&nbsp; Confidences<br><span style="font-weight:normal">(training wind, nominal plant)</span>', group=True)
c1 = d.box(800, 70, 260, 90, '60 LQR flights &rarr; <i>c</i><sub>LQR</sub>:<br>logistic estimate of<br>P(<i>V</i><sub><i>k</i>+<i>H</i></sub> &lt; <i>V</i><sub><i>k</i></sub>), '
           '<i>V</i> = <i>e</i><sup>T</sup><i>Pe</i>, <i>H</i> = 20', 'lqr')
c2 = d.box(800, 180, 260, 110, '40 student flights (&alpha; = 1) &rarr; <i>c</i><sub>S</sub>:<br>soft-label logistic regression of<br>'
           '<i>s<sub>k</sub></i> = exp(&minus;(<i>E</i><sub>20</sub>(<i>k</i>)/&epsilon;<sub>p</sub>)<sup>2</sup>),<br><i>E</i><sub>20</sub>: RMS position error, last 20 steps', 'student')
o1 = d.box(780, 370, 300, 62, '<b>Deployed controller P</b> (Fig. 2):<br>sat(sat(<i>u</i><sub>LQR</sub>) + &alpha; <i>W</i>*&phi;), &alpha; = <i>c</i><sub>S</sub> <i>g</i><sub>L</sub>(<i>c</i><sub>LQR</sub>)', 'out')
d.edge(b6, c2, extra='exitX=1;exitY=0.5;entryX=0;entryY=0.75;', points=[(770, 401), (770, 262)])
d.edge(c1, o1, extra='exitX=1;exitY=0.5;entryX=1;entryY=0.5;', points=[(1095, 115), (1095, 401)])
d.edge(c2, o1)
d.write(os.path.join(OUTDIR, 'Figure_1.drawio'))

# ---------------- Fig. deployed controller ----------------
g = Diagram('controller')
x0 = g.box(10, 120, 120, 70, 'state <i>x<sub>k</sub></i>,<br>reference <i>x</i><sub>ref</sub>, <i>u</i><sub>ref</sub>')
mem = g.box(10, 290, 120, 60, 'memory<br>(<i>x</i><sub><i>k</i>&minus;1</sub>, <i>u</i><sub><i>k</i>&minus;1</sub>)')
lqr = g.box(190, 20, 210, 60, 'LQR (Bryson rule, hover model)<br><i>u</i><sub>LQR</sub> = <i>u</i><sub>h</sub> &minus; <i>K e<sub>k</sub></i>', 'lqr')
fe = g.box(190, 120, 210, 90, 'features &phi;<sub><i>k</i></sub> (27):<br><i>e<sub>k</sub></i>, ' + FH + '<sub><i>k</i></sub>, '
           '&#363;<sub><i>k</i></sub>, &#363;<sub><i>k</i>+5</sub> &minus; &#363;<sub><i>k</i></sub>,<br>&#363;<sub><i>k</i>+20</sub> &minus; &#363;<sub><i>k</i></sub>&nbsp; '
           '(&#363; = <i>u</i><sub>ref</sub> &minus; <i>u</i><sub>h</sub>)', 'student')
fh = g.box(190, 280, 210, 80, FH + '<sub><i>k</i></sub>: external force over the last<br>step from <i>x</i><sub><i>k</i>&minus;1</sub>, <i>u</i><sub><i>k</i>&minus;1</sub>, <i>x<sub>k</sub></i><br>(trapezoidal rule, nominal model)', 'student')
res = g.box(460, 120, 230, 72, 'residual &Delta;<i>u</i> = <i>W</i>*&phi;<sub><i>k</i></sub><br>(<i>W</i>* &isin; &#8477;<sup>4&times;27</sup>, no bias)', 'student')
cs = g.box(460, 280, 230, 60, '<i>c</i><sub>S</sub>(<i>e<sub>k</sub></i>, &Vert;' + FH + '<sub><i>k</i></sub>&Vert;, &Vert;&#363;<sub><i>k</i></sub>&Vert;): logistic', 'student')
cl = g.box(460, 380, 230, 60, '<i>c</i><sub>LQR</sub>(<i>e<sub>k</sub></i>): logistic<br><i>g</i><sub>L</sub> = clip((0.7 &minus; <i>c</i><sub>LQR</sub>)/0.4, 0, 1)', 'lqr')
al = g.box(740, 320, 130, 60, '&alpha; = <i>c</i><sub>S</sub> &middot; <i>g</i><sub>L</sub>')
sat1 = g.box(460, 25, 230, 50, 'sat(<i>u</i><sub>LQR</sub>)', 'lqr')
mul = g.box(740, 131, 130, 50, '&alpha; &Delta;<i>u</i>')
add = g.box(740, 25, 130, 50, 'sat( &middot; + &middot; )')
out = g.box(920, 25, 120, 50, '<i>u<sub>k</sub></i> &rarr; plant', 'out')
g.edge(x0, lqr, extra='exitX=1;exitY=0.25;entryX=0;entryY=0.5;', points=[(160, 137), (160, 50)])
g.edge(x0, fe, extra='exitX=1;exitY=0.5;entryX=0;entryY=0.5;')
g.edge(mem, fh, extra='exitX=1;exitY=0.5;entryX=0;entryY=0.5;')
g.edge(fh, fe, extra='exitX=0.5;exitY=0;entryX=0.5;entryY=1;')
g.edge(x0, cl, extra='exitX=1;exitY=0.85;entryX=0;entryY=0.5;', points=[(170, 180), (170, 410)])
g.edge(fe, cs, extra='exitX=1;exitY=0.8;entryX=0;entryY=0.5;', points=[(430, 192), (430, 310)])
g.edge(fe, res, extra='exitX=1;exitY=0.4;entryX=0;entryY=0.5;')
g.edge(lqr, sat1)
g.edge(cs, al, extra='exitX=1;exitY=0.5;entryX=0;entryY=0.25;', points=[(715, 310), (715, 335)])
g.edge(cl, al, extra='exitX=1;exitY=0.5;entryX=0;entryY=0.75;', points=[(715, 410), (715, 365)])
g.edge(res, mul, extra='exitX=1;exitY=0.5;entryX=0;entryY=0.5;')
g.edge(al, mul, extra='exitX=0.5;exitY=0;entryX=0.5;entryY=1;')
g.edge(sat1, add)
g.edge(mul, add, extra='exitX=0.5;exitY=0;entryX=0.5;entryY=1;')
g.edge(add, out)
g.edge(out, mem, label='store', extra='dashed=1;exitX=0.5;exitY=1;entryX=0.5;entryY=1;', points=[(980, 470), (70, 470)])
g.write(os.path.join(OUTDIR, 'Figure_2.drawio'))
print('ok')
