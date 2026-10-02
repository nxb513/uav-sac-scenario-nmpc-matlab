# DIAGNOSTIC only (not part of the method). gdb Python script run by tools/ci/diag_watchdog.sh
# on a stalled MATLAB process: finds the thread inside HPIPM (d_ocp_qp_ipm_solve) and prints
# the IPM iteration counter and step data, the QP dimensions, the Riccati stage and the
# BLASFEO call it is in. Needs acados built with -g; values the compiler optimized out are
# reported as such. Comparing snapshots taken some seconds apart shows whether the IPM loop
# advances (kk, mu, alpha change) or a single linear-algebra call never returns.
import gdb

gdb.execute('set pagination off')
gdb.execute('set print pretty on')


def stack(th):
    th.switch()
    out = []
    f = gdb.newest_frame()
    while f is not None and len(out) < 60:
        out.append(f)
        try:
            f = f.older()
        except gdb.error:
            break
    return out


def run(cmd):
    try:
        gdb.execute(cmd)
    except gdb.error as e:
        print('  (%s -> %s)' % (cmd, e))


def at(fr, name, cmds):
    for f in fr:
        if f.name() == name:
            f.select()
            print('==== frame %s' % name)
            for c in cmds:
                run(c)
            return
    print('==== frame %s not on the stack' % name)


target = None
for th in gdb.selected_inferior().threads():
    try:
        fr = stack(th)
    except gdb.error:
        continue
    if any(f.name() == 'd_ocp_qp_ipm_solve' for f in fr):
        target = (th, fr)
        break

if target is None:
    print('DIAG: no thread inside d_ocp_qp_ipm_solve')
    run('thread apply all bt 25')
else:
    th, fr = target
    th.switch()
    print('DIAG: thread %d (%s)' % (th.num, th.name))
    run('bt 30')
    dims = ['p *qp->dim'] + ['p *qp->dim->%s@(qp->dim->N+1)' % d
                             for d in ('nx', 'nu', 'nb', 'ng', 'ns', 'nbx', 'nbu')]
    at(fr, 'd_ocp_qp_ipm_solve', ['info args', 'info locals', 'p *arg',
                                  'p *ws->core_workspace', 'p ws->iter', 'p ws->status'] + dims)
    at(fr, 'd_ocp_qp_ipm_delta_step', ['info args', 'info locals'])
    at(fr, 'd_ocp_qp_fact_solve_kkt_step', ['info args', 'info locals'])
    at(fr, 'blasfeo_hp_dgemm_nt', ['info args', 'info locals',
                                   'p *sA', 'p *sB', 'p *sC', 'p *sD'])
    at(fr, 'ocp_qp_xcond_solve', ['info locals'])
    at(fr, 'ocp_nlp_sqp', ['info locals', 'p nlp_mem->iter'])
