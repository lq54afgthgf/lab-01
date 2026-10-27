import re

from dataclasses import dataclass
from findex.types import Index

class QueryNode:
    def evaluate(self, index: Index) -> set[int]:
        raise NotImplementedError

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
        all_docs: set[int] = set(index.doc_meta.keys())
        return all_docs - self.child.evaluate(index)

@dataclass(frozen=True)
class Phrase(QueryNode):
    terms: tuple[str, ...]
    def evaluate(self, index: Index) -> set[int]:
        if not self.terms: return set()
        for t in self.terms:
            if t not in index: return set()

        sets_list: list[set[int]] = [{p.doc_id for p in index[t]} for t in self.terms]
        if not sets_list:
            return set()

        docs: set[int] = set(sets_list[0]).intersection(*sets_list[1:])
        valid_docs: set[int] = set()

        for doc_id in docs:
            pos_lists: list[tuple[int, ...]] = []
            for t in self.terms:
                for p in index[t]:
                    if p.doc_id == doc_id:
                        pos_lists.append(p.positions or tuple())
                        break

            if not pos_lists:
                continue

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

def parse_query(query: str) -> QueryNode:
    tokens = re.findall(r'"[^"]*"|\(|\)|AND|OR|NOT|[^\s()"]+', query)
    pos = 0

    def peek() -> str | None:
        return tokens[pos] if pos < len(tokens) else None

    def consume() -> str:
        nonlocal pos
        if pos >= len(tokens):
            return ""
        t = tokens[pos]
        pos += 1
        return t

    def parse_or() -> QueryNode:
        node = parse_and()
        while peek() == 'OR':
            consume()
            node = node | parse_and()
        return node

    def parse_and() -> QueryNode:
        node = parse_not()
        while peek() and peek() not in ('OR', ')'):
            if peek() == 'AND':
                consume()
            node = node & parse_not()
        return node

    def parse_not() -> QueryNode:
        if peek() == 'NOT':
            consume()
            return ~parse_atom()
        return parse_atom()

    def parse_atom() -> QueryNode:
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
