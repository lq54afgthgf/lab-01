import re
from dataclasses import dataclass
from findex.types import Index

class QueryNode:
    def evaluate(self, index: Index) -> set[int]:
        raise NotImplementedError

    # Перевантаження операторів для зручності (наприклад: Term("a") | Term("b"))
    def __and__(self, other: 'QueryNode') -> 'QueryNode':
        return And(self, other)

    def __or__(self, other: 'QueryNode') -> 'QueryNode':
        return Or(self, other)

    def __invert__(self) -> 'QueryNode':
        return Not(self)

@dataclass(frozen=True)
class Term(QueryNode):
    term: str
    def evaluate(self, index: Index) -> set[int]:
        if self.term not in index:
            return set()
        return {p.doc_id for p in index[self.term]}

@dataclass(frozen=True)
class And(QueryNode):
    left: QueryNode
    right: QueryNode
    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) & self.right.evaluate(index)

@dataclass(frozen=True)
class Or(QueryNode):
    left: QueryNode
    right: QueryNode
    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) | self.right.evaluate(index)

@dataclass(frozen=True)
class Not(QueryNode):
    child: QueryNode
    def evaluate(self, index: Index) -> set[int]:
        all_docs = set(index.doc_meta.keys())
        return all_docs - self.child.evaluate(index)

@dataclass(frozen=True)
class Phrase(QueryNode):
    terms: tuple[str, ...]
    def evaluate(self, index: Index) -> set[int]:
        if not self.terms: return set()
        for t in self.terms:
            if t not in index: return set()

        # Знаходимо спільні документи для всіх слів фрази
        docs = set.intersection(*[{p.doc_id for p in index[t]} for t in self.terms])
        valid_docs = set()

        for doc_id in docs:
            # Збираємо позиції кожного слова
            pos_lists = []
            for t in self.terms:
                for p in index[t]:
                    if p.doc_id == doc_id:
                        pos_lists.append(p.positions or tuple())
                        break

            # Шукаємо послідовність (pos, pos+1, pos+2, ...)
            for start_pos in pos_lists[0]:
                is_match = True
                for i in range(1, len(self.terms)):
                    if (start_pos + i) not in pos_lists[i]:
                        is_match = False
                        break
                if is_match:
                    valid_docs.add(doc_id)
                    break

        return valid_docs

# --- Рекурсивний спуск (Recursive Descent Parser) ---

def parse_query(query: str) -> QueryNode:
    # Токенізація: розбиваємо на фрази в лапках, дужки та окремі слова
    tokens = re.findall(r'"[^"]*"|\(|\)|AND|OR|NOT|[^\s()"]+', query)
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def consume():
        nonlocal pos
        t = tokens[pos]
        pos += 1
        return t

    def parse_or():
        node = parse_and()
        while peek() == 'OR':
            consume()
            node = node | parse_and()
        return node

    def parse_and():
        node = parse_not()
        # Неявний AND: якщо наступний токен не OR і не закриваюча дужка
        while peek() and peek() not in ('OR', ')'):
            if peek() == 'AND':
                consume()
            node = node & parse_not()
        return node

    def parse_not():
        if peek() == 'NOT':
            consume()
            return ~parse_atom()
        return parse_atom()

    def parse_atom():
        t = consume()
        if t == '(':
            node = parse_or()
            if peek() == ')':
                consume()
            return node
        elif t.startswith('"') and t.endswith('"'):
            words = tuple(t.strip('"').lower().split())
            return Phrase(words)
        else:
            return Term(t.lower())

    if not tokens:
        return Term("")
    return parse_or()
