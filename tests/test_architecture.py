from __future__ import annotations

import ast
import graphlib
import hashlib
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src" / "bifurx"
TESTS_ROOT = REPO_ROOT / "tests"
HOMOTOPY_ROOT = SRC_ROOT / "homotopy"


def _iter_py_files(root: pathlib.Path) -> list[pathlib.Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _collect_identifiers(tree: ast.AST, *, skip_node: ast.AST | None = None) -> set[str]:
    """Collect identifier references while skipping `skip_node` and `__all__` string literals."""
    refs: set[str] = set()

    class _Visitor(ast.NodeVisitor):
        def visit(self, node: ast.AST) -> None:
            if node is skip_node:
                return
            super().visit(node)

        def visit_Name(self, node: ast.Name) -> None:
            refs.add(node.id)

        def visit_Attribute(self, node: ast.Attribute) -> None:
            refs.add(node.attr)
            self.generic_visit(node)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            for alias in node.names:
                refs.add(alias.name)

    _Visitor().visit(tree)
    return refs


def _strip_docstring_and_normalize(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Produce a structurally normalized AST dump ignoring names, docstrings, and locations."""
    import copy

    cloned = copy.deepcopy(node)
    if (
        cloned.body
        and isinstance(cloned.body[0], ast.Expr)
        and isinstance(cloned.body[0].value, ast.Constant)
        and isinstance(cloned.body[0].value.value, str)
    ):
        cloned.body = cloned.body[1:]

    cloned.name = "_fn"
    name_map: dict[str, str] = {}

    def _alias(raw: str) -> str:
        if raw not in name_map:
            name_map[raw] = f"_v{len(name_map)}"
        return name_map[raw]

    class _Normalizer(ast.NodeTransformer):
        def visit_arg(self, node: ast.arg) -> ast.arg:
            node.arg = _alias(node.arg)
            node.annotation = None
            return node

        def visit_Name(self, node: ast.Name) -> ast.Name:
            node.id = _alias(node.id)
            return node

        def _visit_fn(
            self, node: ast.FunctionDef | ast.AsyncFunctionDef
        ) -> ast.FunctionDef | ast.AsyncFunctionDef:
            if node is not cloned:
                node.name = _alias(node.name)
            node.returns = None
            self.generic_visit(node)
            return node

        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
            return self._visit_fn(node)  # type: ignore[return-value]

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
            return self._visit_fn(node)  # type: ignore[return-value]

    normalized = _Normalizer().visit(cloned)
    return ast.dump(normalized, include_attributes=False)


def _count_statements(node: ast.AST) -> int:
    return sum(1 for n in ast.walk(node) if isinstance(n, ast.stmt))


def test_homotopy_package_independence() -> None:
    """Verify `bifurx.homotopy` depends only on stdlib, JAX, and itself."""
    allowed_top = sys.stdlib_module_names | {"jax", "bifurx"}
    py_files = _iter_py_files(HOMOTOPY_ROOT)
    violations: list[str] = []

    for file_path in py_files:
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        rel_path = file_path.relative_to(REPO_ROOT)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top not in allowed_top:
                        violations.append(
                            f"{rel_path}:{node.lineno}: forbidden import '{alias.name}'"
                        )
                    elif top == "bifurx" and not (
                        alias.name == "bifurx.homotopy" or alias.name.startswith("bifurx.homotopy.")
                    ):
                        violations.append(
                            f"{rel_path}:{node.lineno}: homotopy cannot import '{alias.name}'"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    anchor = file_path.parent
                    for _ in range(node.level - 1):
                        anchor = anchor.parent
                    if HOMOTOPY_ROOT not in (anchor, *anchor.parents):
                        violations.append(
                            f"{rel_path}:{node.lineno}: relative import escapes homotopy package"
                        )
                    continue

                if node.module is None:
                    continue
                top = node.module.split(".")[0]
                if top not in allowed_top:
                    violations.append(
                        f"{rel_path}:{node.lineno}: forbidden import from '{node.module}'"
                    )
                elif top == "bifurx":
                    if node.module == "bifurx":
                        for alias in node.names:
                            if alias.name != "homotopy":
                                violations.append(
                                    f"{rel_path}:{node.lineno}: cannot import '{alias.name}' from bifurx"
                                )
                    elif not (
                        node.module == "bifurx.homotopy"
                        or node.module.startswith("bifurx.homotopy.")
                    ):
                        violations.append(
                            f"{rel_path}:{node.lineno}: homotopy cannot import from '{node.module}'"
                        )

    assert not violations, "Homotopy independence violations:\n" + "\n".join(violations)


def test_no_unused_functions_or_classes() -> None:
    """Verify no module-level function or class in `src/bifurx/` is dead code."""
    src_files = _iter_py_files(SRC_ROOT)
    test_files = _iter_py_files(TESTS_ROOT)

    src_trees = {p: ast.parse(p.read_text(encoding="utf-8"), filename=str(p)) for p in src_files}
    test_trees = {p: ast.parse(p.read_text(encoding="utf-8"), filename=str(p)) for p in test_files}

    test_refs: set[str] = set()
    for tree in test_trees.values():
        test_refs |= _collect_identifiers(tree)

    non_init_src_refs = {
        p: _collect_identifiers(tree) for p, tree in src_trees.items() if p.name != "__init__.py"
    }

    unused: list[str] = []
    for path, tree in src_trees.items():
        rel_path = path.relative_to(REPO_ROOT)
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            name = node.name
            if name.startswith("__") and name.endswith("__"):
                continue

            same_file_external = _collect_identifiers(tree, skip_node=node)
            other_src_refs = set().union(
                *(refs for sp, refs in non_init_src_refs.items() if sp != path)
            )

            if name.startswith("_"):
                if name not in same_file_external and name not in other_src_refs:
                    unused.append(f"{rel_path}:{node.lineno}: unused private '{name}'")
            else:
                if (
                    name not in same_file_external
                    and name not in other_src_refs
                    and name not in test_refs
                ):
                    unused.append(f"{rel_path}:{node.lineno}: unused public '{name}'")

    assert not unused, "Unused definitions detected:\n" + "\n".join(unused)


def test_no_duplicate_definitions() -> None:
    """Verify no duplicate top-level symbol names or copy-pasted functions exist in `src/bifurx/`."""
    src_files = _iter_py_files(SRC_ROOT)
    name_locations: dict[str, list[str]] = {}
    body_hashes: dict[str, list[str]] = {}

    for path in src_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel_path = path.relative_to(REPO_ROOT)

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not (node.name.startswith("__") and node.name.endswith("__")):
                    name_locations.setdefault(node.name, []).append(f"{rel_path}:{node.lineno}")

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if _count_statements(node) < 10:
                    continue
                digest = hashlib.sha256(_strip_docstring_and_normalize(node).encode()).hexdigest()
                body_hashes.setdefault(digest, []).append(f"{rel_path}:{node.lineno} ({node.name})")

    dup_names = {k: v for k, v in name_locations.items() if len(v) > 1}
    dup_bodies = {k: v for k, v in body_hashes.items() if len(v) > 1}

    assert not dup_names, f"Duplicate top-level definitions found: {dup_names}"
    assert not dup_bodies, f"Structurally duplicate functions found: {list(dup_bodies.values())}"


def _path_to_module(path: pathlib.Path) -> str:
    rel = path.relative_to(SRC_ROOT.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_internal_imports(
    file_path: pathlib.Path,
    tree: ast.AST,
    known_modules: set[str],
) -> set[str]:
    """Collect all internal `bifurx` module dependencies (runtime, local, and TYPE_CHECKING)."""
    mod_name = _path_to_module(file_path)
    is_pkg = file_path.name == "__init__.py"
    pkg_parts = mod_name.split(".") if is_pkg else mod_name.split(".")[:-1]
    deps: set[str] = set()

    def _add_target(candidate: str) -> None:
        if candidate in known_modules and candidate != mod_name:
            deps.add(candidate)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _add_target(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                base_parts = pkg_parts[: len(pkg_parts) - (node.level - 1)]
                base = ".".join(base_parts)
                target = f"{base}.{node.module}" if node.module else base
            else:
                target = node.module or ""

            if not target.startswith("bifurx"):
                continue

            _add_target(target)
            for alias in node.names:
                _add_target(f"{target}.{alias.name}")

    return deps


def test_internal_import_graph_is_dag() -> None:
    """Verify internal module imports across `src/bifurx/` form a strict DAG (zero cycles)."""
    src_files = _iter_py_files(SRC_ROOT)
    known_modules = {_path_to_module(p) for p in src_files}
    graph: dict[str, set[str]] = {}

    for path in src_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        mod = _path_to_module(path)
        graph[mod] = _resolve_internal_imports(path, tree, known_modules)

    try:
        list(graphlib.TopologicalSorter(graph).static_order())
    except graphlib.CycleError as exc:
        cycle = " -> ".join(exc.args[1])
        raise AssertionError(f"Circular import detected in src/bifurx/: {cycle}") from exc
