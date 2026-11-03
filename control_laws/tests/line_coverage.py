"""
Which lines of the three optimal diagonal laws do the golden scenarios execute?

A refactor is only as safe as the tests that run the code it moves, and golden_optdiag
only proves what its scenarios reach.  This runs them under a line tracer (sys.settrace:
`coverage` is not installed, and Python 3.11 has no sys.monitoring) and lists, per
function, the executable lines that never ran.  About 90 s for the full set.

    python control_laws/tests/line_coverage.py                  summary by function
    python control_laws/tests/line_coverage.py --lines          ...with each missing line's source
    python control_laws/tests/line_coverage.py --only NAME ...  just those scenarios

Docstrings and comments are not executable and are not counted.
"""
import argparse
import ast
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
LAW_DIR = os.path.dirname(HERE)
FILES = ['optimal_diagonal_control.py', 'optimal_diagonal_control_fast.py',
         'octave_band_switching_control.py']
TARGETS = {os.path.join(LAW_DIR, f): f for f in FILES}
hit = {f: set() for f in FILES}


def _local(frame, event, arg):
    if event == 'line':
        hit[TARGETS[frame.f_code.co_filename]].add(frame.f_lineno)
    return _local


def _global(frame, event, arg):
    f = TARGETS.get(frame.f_code.co_filename)
    if f is None:
        return None
    hit[f].add(frame.f_lineno)
    return _local


def executable_lines(path):
    src = open(path).read()
    tree = ast.parse(src)
    doc = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
            b = node.body
            if (b and isinstance(b[0], ast.Expr) and isinstance(getattr(b[0], 'value', None), ast.Constant)
                    and isinstance(b[0].value.value, str)):
                doc.update(range(b[0].lineno, b[0].end_lineno + 1))
    lines = set()

    def walk(code):
        for _, _, ln in code.co_lines():
            if ln:
                lines.add(ln)
        for k in code.co_consts:
            if isinstance(k, types.CodeType):
                walk(k)
    walk(compile(src, path, 'exec'))
    return lines - doc, tree


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--lines', action='store_true', help='print the source of each line that never ran')
    ap.add_argument('--only', nargs='+', metavar='NAME', help='run only these scenarios')
    a = ap.parse_args()
    sys.settrace(_global)                       # before the laws are loaded: module-level lines count
    sys.path.insert(0, HERE)
    import golden_optdiag as g
    laws, plants = g.load_laws(), {}
    for sc in g.SCENARIOS:
        if not a.only or sc['name'] in a.only:
            g.run_scenario(sc, laws, plants)
    sys.settrace(None)
    for f in FILES:
        ex, tree = executable_lines(os.path.join(LAW_DIR, f))
        src = open(os.path.join(LAW_DIR, f)).read().split('\n')
        funcs = sorted(((n.lineno, n.end_lineno, n.name) for n in ast.walk(tree)
                        if isinstance(n, ast.FunctionDef)))

        def owner(ln):
            best = None
            for lo, hi, name in funcs:
                if lo <= ln <= hi and (best is None or lo > best[0]):
                    best = (lo, name)
            return best[1] if best else '<module>'
        missing = sorted(ex - hit[f])
        by = {}
        for ln in missing:
            by.setdefault(owner(ln), []).append(ln)
        print(f'\n{f}: {len(ex) - len(missing)} of {len(ex)} executable lines run '
              f'({100*(len(ex) - len(missing))/len(ex):.0f}%)')
        for name, lines in sorted(by.items(), key=lambda kv: -len(kv[1])):
            print(f'   {name:34s} {len(lines):3d} never run (first L{lines[0]})')
            if a.lines:
                for ln in lines:
                    print(f'        L{ln}: {src[ln-1].strip()[:110]}')


if __name__ == '__main__':
    main()
