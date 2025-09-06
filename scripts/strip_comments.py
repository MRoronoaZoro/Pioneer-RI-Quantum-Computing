import os
import ast
import io
import tokenize

ROOT = r"c:\Users\59415\Desktop\Pioneer RI\Research\Code"


def get_docstring_ranges(src):
    try:
        module = ast.parse(src)
    except Exception:
        return None, set()

    keep_doc_ranges = set()
    module_doc_range = None


    if module.body and isinstance(module.body[0], ast.Expr) and isinstance(module.body[0].value, ast.Constant) and isinstance(module.body[0].value.value, str):
        node = module.body[0]
        module_doc_range = (node.lineno, getattr(node, 'end_lineno', node.lineno))


    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                n = node.body[0]
                keep_doc_ranges.add((n.lineno, getattr(n, 'end_lineno', n.lineno)))

    return module_doc_range, keep_doc_ranges


def strip_comments_preserve_func_docstrings(path):
    with open(path, 'r', encoding='utf-8') as f:
        src = f.read()

    module_doc_range, keep_doc_ranges = get_docstring_ranges(src)


    lines = src.splitlines()
    if module_doc_range:
        start_ln = module_doc_range[0] - 1
        end_ln = module_doc_range[1] - 1
        for ln in range(start_ln, end_ln + 1):
            if 0 <= ln < len(lines):
                lines[ln] = ''
    pre_src = '\n'.join(lines)
    if src.endswith('\n'):
        pre_src += '\n'


    out_tokens = []
    try:
        tokgen = tokenize.generate_tokens(io.StringIO(pre_src).readline)
    except Exception:
        return False, 'tokenize-error'

    for tok in tokgen:
        tok_type, tok_str, start, end, line = tok
        if tok_type == tokenize.COMMENT:
            continue

        out_tokens.append(tok)

    new_src = tokenize.untokenize(out_tokens)


    new_src = '\n'.join(ln.rstrip() for ln in new_src.splitlines())
    if pre_src.endswith('\n') and not new_src.endswith('\n'):
        new_src += '\n'

    if new_src != src:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_src)
        return True, 'changed'
    else:
        return False, 'unchanged'


ess_skipped_dirs = {'results', 'results_CTQW_1', 'results_NoOracle_1', 'results_Oracle_1.4', '.git', '.venv', 'venv', '__pycache__', 'datasets'}

changed_files = []
errors = []

for dirpath, dirnames, filenames in os.walk(ROOT):

    bn = os.path.basename(dirpath)
    if bn in ess_skipped_dirs:
        continue
    if 'datasets' in dirpath.replace('\\', '/').lower():
        continue

    for fn in filenames:
        if not fn.endswith('.py'):
            continue
        path = os.path.join(dirpath, fn)
        changed, status = strip_comments_preserve_func_docstrings(path)
        if changed:
            changed_files.append(path)
        elif status == 'tokenize-error':
            errors.append(path)


extra_candidates = [os.path.join(ROOT, 'CTQW')]
for candidate in extra_candidates:
    if os.path.isfile(candidate):
        changed, status = strip_comments_preserve_func_docstrings(candidate)
        if changed and candidate not in changed_files:
            changed_files.append(candidate)
        elif status == 'tokenize-error' and candidate not in errors:
            errors.append(candidate)

print('Changed:', len(changed_files))
for p in changed_files:
    print(p)
if errors:
    print('Errors:', len(errors))
    for p in errors:
        print('ERR', p)