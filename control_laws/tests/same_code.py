"""
Is the code of two versions of a file identical once comments and docstrings are
ignored?  The proof that a documentation-only change did not touch behaviour: comments
are not in the syntax tree, and docstrings are removed from it before comparing.

    python control_laws/tests/same_code.py OLD.py NEW.py
    python control_laws/tests/same_code.py --rev HEAD~1 control_laws/optimal_diagonal_control.py
        (compares that file at the git revision with the working copy)

Exit code 0 if identical, 1 if the code differs.  Any string that is not a docstring
(a Param description, a log message) is code and must match.
"""
import ast
import subprocess
import sys


def stripped_dump(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], 'value', None), ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree)


def main(argv):
    if len(argv) == 4 and argv[0] == '--rev':
        old = subprocess.run(['git', 'show', f'{argv[1]}:{argv[2]}'], capture_output=True, text=True, check=True).stdout
        new = open(argv[3] if len(argv) > 3 else argv[2]).read()
        label = f'{argv[2]} at {argv[1]} vs the working copy'
    elif len(argv) == 3 and argv[0] == '--rev':
        old = subprocess.run(['git', 'show', f'{argv[1]}:{argv[2]}'], capture_output=True, text=True, check=True).stdout
        new = open(argv[2]).read()
        label = f'{argv[2]} at {argv[1]} vs the working copy'
    elif len(argv) == 2:
        old, new = open(argv[0]).read(), open(argv[1]).read()
        label = f'{argv[0]} vs {argv[1]}'
    else:
        print(__doc__)
        return 2
    same = stripped_dump(old) == stripped_dump(new)
    print(('IDENTICAL' if same else 'DIFFERENT') + f' code (comments and docstrings aside): {label}')
    return 0 if same else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
