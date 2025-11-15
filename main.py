
import argparse
import json
import sys
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional
import urllib.request
import urllib.error
import re
import textwrap
import io

def parse_args():
    p = argparse.ArgumentParser(description="Dependency graph visualizer (Variant 1).")
    p.add_argument("--package", "-p", required=True, help="Имя анализируемого пакета")
    p.add_argument("--repo-url", help="URL репозитория (для PyPI укажите https://pypi.org)")
    p.add_argument("--repo-file", help="Путь к файлу тестового репозитория")
    p.add_argument("--test-repo", action="store_true", help="Включить режим тестового репозитория (использует --repo-file)")
    p.add_argument("--version", "-v", help="Версия пакета (для реального репозитория)")
    p.add_argument("--max-depth", "-m", type=int, default=5, help="Максимальная глубина анализа зависимостей (default=5)")
    p.add_argument("--ascii-tree", action="store_true", help="Вывести зависимости в виде ASCII-дерева")
    p.add_argument("--plantuml", action="store_true", help="Сформировать текст PlantUML и вывести")
    p.add_argument("--print-order", action="store_true", help="Вывести порядок загрузки зависимостей (topological-like)")
    p.add_argument("--show-direct", action="store_true", help="Только прямые зависимости и выход (этап 2)")
    return p.parse_args()

# Этап 1: при запуске — вывести все параметры (key=value) и проверка ошибок

def print_parameters(args):
    print("Запущено с параметрами:")
    for k, v in sorted(vars(args).items()):
        print(f"  {k} = {v}")
    # Проверки параметров:
    if args.test_repo and not args.repo_file:
        print("Ошибка: режим --test-repo требует --repo-file (путь к файлу тестового репозитория).", file=sys.stderr)
        sys.exit(1)
    if not args.test_repo and not args.repo_url:
        print("Предупреждение: вы не указали --repo-url и не включили --test-repo. Невозможно получить данные для реального репозитория.", file=sys.stderr)


# Этап 2: сбор данных
# - Для тестового репо: читаем файл и парсим граф
# - Для реального PyPI: используем PyPI JSON API чтобы получить requires_dist

def load_test_repo(path: str) -> Dict[str, List[str]]:
    """
    Читает тестовый файл в формате:
    A: B C
    B: D
    C:
    Возвращает словарь package -> list(deps)
    """
    repo = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" not in line:
                    raise ValueError(f"Неправильная строка на {i}: '{line}' (ожидается 'A: B C')")
                pkg, deps = line.split(":", 1)
                pkg = pkg.strip()
                deps_list = [d for d in (deps.strip().split() if deps.strip() else [])]
                repo[pkg] = deps_list
        return repo
    except FileNotFoundError:
        print(f"Ошибка: файл тестового репозитория не найден: {path}", file=sys.stderr)
        sys.exit(1)

def get_direct_deps_from_pypi(package: str, version: Optional[str]) -> List[str]:
    """
    Использует PyPI API: https://pypi.org/pypi/{package}/{version}/json
    Если version None — берет latest.
    Возвращает список зависимостей (requirement names, без extras/versions if possible).
    """
    if version:
        url = f"https://pypi.org/pypi/{package}/{version}/json"
    else:
        url = f"https://pypi.org/pypi/{package}/json"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        print(f"HTTPError при обращении к PyPI: {e}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Ошибка при обращении к PyPI: {e}", file=sys.stderr)
        return []

    # В metadata/ info есть requires_dist — список строк вида "packagename (>=1.2); extra == '...'"
    info = data.get("info", {})
    requires = info.get("requires_dist") or []
    deps = []
    for r in requires:
        # извлечь имя пакета из requirement
        # требование может содержать extras и версии: "requests (>=2.0); python_version < '3.0'"
        # возьмём первую часть до пробела, скобки или ';'
        # более надёжно — regex
        m = re.match(r"^\s*([A-Za-z0-9_\-\.]+)", r)
        if m:
            deps.append(m.group(1))
    # Удаляем duplicates
    seen = set()
    out = []
    for d in deps:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


# Этап 3: получение графа зависимостей (DFS рекурсией), макс глубина, обработка циклов

class DependencyGraph:
    def __init__(self):
        # adjacency: package -> list of dependencies
        self.adj: Dict[str, List[str]] = defaultdict(list)

    def add_edges(self, pkg: str, deps: List[str]):
        self.adj[pkg].extend(deps)

    def get_direct(self, pkg: str) -> List[str]:
        return self.adj.get(pkg, [])

    def build_from_test_repo(self, repo: Dict[str, List[str]]):
        for k, v in repo.items():
            self.adj[k] = v[:]  # shallow copy

    def build_from_pypi(self, root_pkg: str, version: Optional[str], max_depth: int = 5):
        """
        Рекурсивно собирает граф, начиная с root_pkg, запрашивая PyPI для каждой новой вершины.
        Ограничивает глубину через max_depth.
        """
        visited = set()

        def dfs(pkg: str, depth: int):
            if depth > max_depth:
                return
            if pkg in visited:
                return
            visited.add(pkg)
            deps = get_direct_deps_from_pypi(pkg, None)  # получение прямых зависимостей
            # Добавляем в граф
            self.add_edges(pkg, deps)
            for d in deps:
                dfs(d, depth + 1)

        dfs(root_pkg, 1)

    def dfs_collect(self, root: str, max_depth: int = 5) -> Tuple[Dict[str, List[str]], List[List[str]]]:
        """
        Собирает граф транзитивно от root до max_depth с рекурсией, возвращая
        (adj_subgraph, cycles)
        cycles — список циклов (каждый как список вершин в порядке).
        """
        sub_adj: Dict[str, List[str]] = defaultdict(list)
        visited: Set[str] = set()
        in_stack: List[str] = []
        cycles: List[List[str]] = []

        def dfs(node: str, depth: int):
            if depth > max_depth:
                return
            if node in in_stack:
                # нашли цикл: извлечём цикл из стека
                idx = in_stack.index(node)
                cycle = in_stack[idx:] + [node]
                cycles.append(cycle)
                return
            if node in visited:
                return
            in_stack.append(node)
            visited.add(node)
            deps = self.get_direct(node)
            sub_adj[node] = deps[:]
            for d in deps:
                dfs(d, depth + 1)
            in_stack.pop()

        dfs(root, 1)
        return sub_adj, cycles

    def ascii_tree(self, root: str, max_depth: int = 5) -> str:
        """
        Возвращает ASCII-дерево зависимостей.
        """
        lines = []
        visited = set()

        def rec(node: str, depth: int, prefix: str):
            if depth > max_depth:
                lines.append(prefix + node + " (max depth reached)")
                return
            if node in visited:
                lines.append(prefix + node + " (visited/cycle)")
                return
            visited.add(node)
            lines.append(prefix + node)
            deps = self.get_direct(node)
            for i, d in enumerate(deps):
                is_last = (i == len(deps) - 1)
                new_prefix = prefix + ("    " if is_last else "│   ")
                branch = "└── " if is_last else "├── "
                rec(d, depth + 1, prefix + branch)

        rec(root, 1, "")
        return "\n".join(lines)

    def loading_order(self, root: str, max_depth: int = 50) -> List[str]:
        """
        Простой DFS-based reverse postorder для порядка загрузки (примерно topological,
        но в графе с циклами он не будет строгим).
        """
        visited = set()
        order = []

        def dfs(node: str, depth: int):
            if depth > max_depth:
                return
            if node in visited:
                return
            visited.add(node)
            for d in self.get_direct(node):
                dfs(d, depth + 1)
            order.append(node)

        dfs(root, 1)
        order.reverse()
        return order

    def plantuml(self, sub_adj: Dict[str, List[str]]) -> str:
        """
        Формирует PlantUML-описание для направленного графа.
        Например:
        @startuml
        digraph {
          A -> B
        }
        @enduml
        """
        out = io.StringIO()
        out.write("@startuml\n")
        out.write("digraph dependencies {\n")
        for a, deps in sub_adj.items():
            if not deps:
                out.write(f'  "{a}";\n')
            for d in deps:
                out.write(f'  "{a}" -> "{d}";\n')
        out.write("}\n")
        out.write("@enduml\n")
        return out.getvalue()


# Утилиты и демонстрация функциональности (печать результатов)

def demo_on_test_repo(args, graph: DependencyGraph):
    print("\n--- Тестовый режим: загружен граф из файла ---")
    # Этап 2: вывести прямые зависимости запрошенного пакета
    pkg = args.package
    if args.show_direct:
        print(f"Прямые зависимости пакета {pkg}: {graph.get_direct(pkg)}")
        return

    sub_adj, cycles = graph.dfs_collect(pkg, max_depth=args.max_depth)
    print(f"\nСубграф (до глубины {args.max_depth}) для пакета {pkg}:")
    for a, deps in sub_adj.items():
        print(f"  {a} -> {deps}")

    if cycles:
        print("\nНайдены циклы:")
        for c in cycles:
            print("  " + " -> ".join(c))
    else:
        print("\nЦиклы не обнаружены.")

    if args.ascii_tree:
        print("\nASCII-дерево зависимостей:")
        print(graph.ascii_tree(pkg, max_depth=args.max_depth))

    if args.print_order:
        order = graph.loading_order(pkg, max_depth=args.max_depth*5)
        print("\nПорядок загрузки зависимостей (приближенный):")
        print("  " + " -> ".join(order))

    if args.plantuml:
        plant = graph.plantuml(sub_adj)
        print("\nPlantUML-представление:")
        print(plant)

def demo_on_pypi(args, graph: DependencyGraph):
    print("\n--- Режим PyPI: собран граф по API ---")
    pkg = args.package
    if args.show_direct:
        deps = graph.get_direct(pkg)
        print(f"Прямые зависимости пакета {pkg}: {deps}")
        return

    sub_adj, cycles = graph.dfs_collect(pkg, max_depth=args.max_depth)
    print(f"\nСубграф (до глубины {args.max_depth}) для пакета {pkg}:")
    for a, deps in sub_adj.items():
        print(f"  {a} -> {deps}")

    if cycles:
        print("\nНайдены циклы:")
        for c in cycles:
            print("  " + " -> ".join(c))
    else:
        print("\nЦиклы не обнаружены.")

    if args.ascii_tree:
        print("\nASCII-дерево зависимостей:")
        print(graph.ascii_tree(pkg, max_depth=args.max_depth))

    if args.print_order:
        order = graph.loading_order(pkg, max_depth=args.max_depth*5)
        print("\nПорядок загрузки зависимостей (приближенный):")
        print("  " + " -> ".join(order))

    if args.plantuml:
        plant = graph.plantuml(sub_adj)
        print("\nPlantUML-представление:")
        print(plant)


# main

def main():
    args = parse_args()
    print_parameters(args)

    graph = DependencyGraph()

    # Этап 2: сбор данных
    if args.test_repo:
        repo = load_test_repo(args.repo_file)
        graph.build_from_test_repo(repo)
        demo_on_test_repo(args, graph)
        return

    # Реальный режим: используем PyPI API (по умолчанию)
    if args.repo_url and "pypi.org" in args.repo_url:
        print("\nПопытка собрать граф через PyPI API (может занять время)...")
        # Собираем рекурсивно (ограничено max_depth)
        # Получаем для корневого пакета его прямые зависимости и рекурсивно
        # Добавляем корневой пакет отдельно (чтобы graph.adj[root] заполнен)
        root_deps = get_direct_deps_from_pypi(args.package, args.version)
        graph.add_edges(args.package, root_deps)
        # Построим рекурсивно (graph.build_from_pypi сам делает dfs)
        graph.build_from_pypi(args.package, args.version, max_depth=args.max_depth)
        demo_on_pypi(args, graph)
        return

    # Если не тест и не pypi-url
    print("Невозможно собрать реальные зависимости: не задан --repo-url (или указанный URL не поддерживается).", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    main()
