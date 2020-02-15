import yaml
from bidict import bidict
import re
import random
from sys import argv

random.seed(1)


yaml_load_options = {
    #'Loader': yaml.CLoader,
}

yaml_dump_options = {
    #'default_flow_style': True,
    #'Dumper': yaml.CDumper,
}

FULL_H = 2500
FULL_V = 2502

DASH_LEFT = 2574
DASH_TOP = 2575
DASH_RIGHT = 2576
DASH_BOTTOM = 2577

def handle_yaml_tree(tree):
    def tup_from_string(s):
        return (int(s[:-1]), s[-1])

    links = {}

    for name, lst in tree['links'].items():
        links[name] = make_cycle((tup_from_string(s) for s in lst))
    
    return links, tree['positive']


def node_print(node, fmt=None):
    prefix = f'{node[0]}' if fmt is None else f'{node[0]:{fmt}}'
    return prefix + node[1]

def Node(s):
    if isinstance(s, str):
        return (int(s[:-1]), s[-1])
    else:
        return s

class Edge(object):
    def __init__(self, diag, p=None, q=None, forward=True):
        if isinstance(p, str):
            forward = p[-1] == '>'
            p = (int(p[:-2]), p[-2:-1])

        self.forward = forward
        if p is not None:
            self.p = p
            self.q = diag.forward(p) if forward else diag.backward(p)
            if q is not None and self.q != q:
                raise ValueError('Invalid edge: {p} does not point to {q}')
        elif q is not None:
            self.q = q
            self.p = diag.backward(q) if forward else diag.forward(q)
        else:
            raise ValueError('Cannot create edge - neither p nor q specified')

    @property
    def data(self):
        return (self.p, self.q, self.forward)

    def __hash__(self):
        return hash((self.p, self.q, self.forward))

    @property
    def start(self):
        return self.p if self.forward else self.q

    @property
    def end(self):
        return self.q if self.forward else self.p

    @property
    def reverse(self):
        return Edge(self.q, self.p, not self.forward)

    # @property
    # def linkname(self):
    #     return self.parent.index[self.p]

    # @property
    # def link(self):
    #     return self.parent.links[self.linkname]

    def __eq__(self, other):
        return self.p == other.p and self.q == other.q and self.forward == other.forward

    def __str__(self):
        arrow = '\u2192' if self.forward else '\u2190'
        return f'({node_print(self.p)}){arrow}({node_print(self.q)})'

    def __repr__(self):
        return f'Edge({self.p}, {self.q}, {self.forward})'



class LinkDiag(object):
    def __init__(self, repr, name=None):
        ''' Four possibilies are accounted for:
            (1) repr is Pre-existing LinkDiag object: convert to "yaml object" then convert back, creating clone of original
            (2) repr is Raw yaml string: use "from_yaml"
            (3) repr is a "yaml object" i.e. the result of running the raw yaml through yaml.load
            (4) (name, repr) is a key/value pair from a "yaml object"
        '''
        if name is not None:
            self._init(name, repr)
            return

        if isinstance(repr, LinkDiag):
            repr = repr._yaml(True)
        elif isinstance(repr, str):
            repr = yaml.load(repr, **yaml_load_options)

        for k, v in repr.items():
            self._init(k, v)
            break

        # self.Edge = edgeClass(self)
    
    def _init(self, k, v):
        self.name = k
        self.links, positive = handle_yaml_tree(v)
        # self.index = {node: name for name, link in self.links.items() for node in link}
        self.index = {node: link for _, link in self.links.items() for node in link}
        self.crossings = {n: (None if sign == '*' else (n in positive)) for (n, sign) in self.index}
        self.cursor = 1 + max(self.crossings)

    # def get_polygons(self):
    #     polygons = {}
    #     for node in self.index:
    #         if node in visited:
    #             break

    #         # Algorithm:
    #         # Go anticlockwise around polygons.
    #         # So if going fwds and encounter (top, positive) then switch and go forwards
    #         #    if going fwds and encounter (top, negative) then switch and go backwards
    #         #    if going fwds and encounter (bottom, positive) then switch and go backwards
    #         #    if going fwds and encounter (bottom, negative) then switch and go forwards

    #         e = self.Edge(node)
    #         current_poly = [e]
    #         polygons[e.p] = current_poly
    #         visited.add(e.p)
    #         current_poly = [e]
    #         fwd_flag = True

    #         while True:
    #             end_crossing, sign = e.q
    #             positive = self.crossings[end_crossing]
    #             opposite_pt = self.node_flip(e.q)
    #             fwd_flag ^= (sign == '-') ^ (not positive)

    #             e = self.Edge(opposite_pt, fwd_flag)
    #             if e.p in visited:
    #                 break

    #             visited.add(e.p)
    #             current_poly.append(e)

    def poly_step(self, edge):
        crossing_num, sign = edge.q
        if sign == '*':
            return Edge(self, edge.q, forward=edge.forward)
        else:
            new_sign = '-' if sign == '+' else '+'
            positive = self.crossings[crossing_num]
            return Edge(self, (crossing_num, new_sign), forward=edge.forward ^ positive ^ (sign=='+'))

    def get_polygons(self):
        visited = set()
        polygons = []

        for node in self.index:
            for e in [Edge(self, node, forward=True), Edge(self, node, forward=False)]:
                if e.data in visited:
                    continue
                current_poly = [e]
                visited.add(e.data)

                while True:
                    e = self.poly_step(e)
                    if e.data in visited:
                        break

                    visited.add(e.data)
                    current_poly.append(e)

                polygons.append(current_poly)

        return polygons


    def add_crossing(self, top, bottom, positive):
        '''Arguments 'top' and 'bottom' should be Edges'''
        # top_linkname = self.index[top.p]
        # bottom_linkname = self.index[bottom.p]
        # top_link = self.links[top_linkname]
        # bottom_link = self.links[bottom_linkname]
        top_link = self.index[top.p]
        bottom_link = self.index[bottom.p]

        n = self.gen_crossing_num()
        top_link[top.start] = (n, '+')
        top_link[(n, '+')] = top.end
        bottom_link[bottom.start] = (n, '-')
        bottom_link[(n, '-')] = bottom.end

        # Book-keeping
        self.crossings[n] = positive
        # self.index[(n, '+')] = top_linkname
        # self.index[(n, '-')] = bottom_linkname
        self.index[(n, '+')] = top_link
        self.index[(n, '-')] = bottom_link

        return n

    def add_dummy(self, edge):
        # linkname = self.index[edge.p]
        # link = self.links[linkname]
        link = self.index[edge.p]

        n = self.gen_crossing_num()
        link[edge.start] = (n, '*')
        link[(n, '*')] = edge.end

        self.crossings[n] = False
        # self.index[(n, '*')] = linkname
        self.index[(n, '*')] = link

        return n

    def del_crossing(self, cnum):
        for sign in ('+', '-'):
            pt = (cnum, sign)
            # linkname = self.index[pt]
            # link = self.links[linkname]
            link = self.index[pt]
            alpha = link[pt]
            del link[pt]
            del self.index[pt]
            link[link.inverse[pt]] = alpha
        del self.crossings[cnum]

    def gen_crossing_num(self):
        cnum = self.cursor
        self.cursor += 1
        return cnum

    def __getitem__(self, node):
        # return self.links[self.index[node]][node]
        return (self.index[node])[node]

    @property
    def positive(self):
        return set(n for n, v in self.crossings.items() if v)

    @property
    def negative(self):
        return set(n for n, v in self.crossings.items() if not v)

    def forward(self, node):
        # return self.links[self.index[node]][node]
        return (self.index[node])[node]

    def backward(self, node):
        # return self.links[self.index[node]].inv[node]
        return self.index[node].inv[node]

    @staticmethod
    def node_flip(node):
        sign = node[1]
        return (node[0], '+' if sign == '-' else '-' if sign == '+' else '*')

    @property
    def yaml(self):
        return self._yaml(False)

    def _yaml(self, as_object):
        links = {}
        foo = {'links': links, 'positive': list(self.positive)}

        for name, d in self.links.items():
            lst = []
            if len(d) > 0:
                for k in d:
                    root = k
                    break
                lst.append(node_print(root))
                curs = d[root]
                while curs != root:
                    lst.append(node_print(curs))
                    curs = d[curs]

            links[name] = lst

        obj_to_dump = {self.name if self.name else '<Anon>': foo}
        return obj_to_dump if as_object else yaml.dump(obj_to_dump, **yaml_dump_options)

    @property
    def wrythe(self):
        return sum(1 if v else -1 for _, v in self.crossings.items())

    # Represent an edge as a triple: (node from, node to, direction).

    def polygon(self, node):
        pass
    
    def traverse(self, node, fwd_flag):
        pass

    def _ensure_edge(self, edge):
        if isinstance(edge, str):
            return Edge(self, edge)
        else:
            return edge

    def reide1(self, edge, pos_flag):
        edge = self._ensure_edge(edge)

        start, end, fwd = edge.start, edge.end, edge.forward
        # linkname = self.index[edge.q]
        # link = self.links[linkname]
        link = self.index[edge.q]
        n = self.gen_crossing_num()

        self.crossings[n] = pos_flag
        # self.index[(n, '+')] = linkname
        # self.index[(n, '-')] = linkname
        self.index[(n, '+')] = link
        self.index[(n, '-')] = link

        sign1, sign2 = ('+', '-') if fwd else ('-', '+')

        if start == end:
            del link[start]
            del self.index[start]
            del self.crossings[start[0]]
            start = (n, sign2)
            end = (n, sign1)

        link[start] = (n, sign1)
        link[(n, sign1)] = (n, sign2)
        link[(n, sign2)] = end

    def reide2(self, top_edge, bottom_edge):
        top_edge = self._ensure_edge(top_edge)
        bottom_edge = self._ensure_edge(bottom_edge)

        self.add_crossing(top_edge, bottom_edge, positive=False)
        e_top = Edge(self, top_edge.q, forward=not top_edge.forward)
        e_bottom = Edge(self, bottom_edge.p, forward=bottom_edge.forward)
        self.add_crossing(e_top, e_bottom, positive=True)

    def invert_edge(self, edge):
        link = self.index[edge.p]
        alpha = self.backward(edge.start)
        omega = self.forward(edge.end)
        del link[alpha]
        del link[edge.start]
        del link[edge.end]
        link[alpha] = edge.end
        link[edge.end] = edge.start
        link[edge.start] = omega

    def reide3(self, edge1, edge2, edge3):
        for e in [edge1, edge2, edge3]:
            f = self._ensure_edge(e)
            self.invert_edge(f)

    def unreide1(self, edge):
        edge = self._ensure_edge(edge)
        # start, end, link = edge.start, edge.end, self.links[self.index[edge.p]]
        start, end, link = edge.start, edge.end, self.index[edge.p]
        alpha = self.backward(start)
        omega = self.forward(end)

        if alpha == end:
            n = self.add_dummy(Edge(self, start, forward=False))
            alpha = (n, '*')
            omega = (n, '*')

        del self.index[start]
        del self.index[end]
        del self.crossings[start[0]]

        del link[start]
        del link[end]
        link[alpha] = omega

    def unreide2(self, edge1, edge2=None):
        edge = self._ensure_edge(edge1)
        '''Only need a single edge to specify 'undo Reidemeister-2' operation'''
        self.del_crossing(edge1.p[0])
        self.del_crossing(edge1.q[0])

    def random_edge(self):
        node = random.choice(list(self.index))
        return Edge(self, node)

    def random_move(self):
        while not self.random_move_fallible():
            pass

    def random_move_fallible(self):
        PROB_TYPE_ONE = 0.1

        if random.random() < PROB_TYPE_ONE:
            e = self.random_edge()
            self.reide1(e, random.randint(0, 1) == 1)
            return True
        else:
            polys = self.get_polygons()
            lengths = [len(lst) for lst in polys]
            onegons = []
            twogons = []
            threegons = []
            manygons = []
            for poly in polys:
                if len(poly) == 1:
                    onegons.append(poly)
                elif len(poly) == 2:
                    twogons.append(poly)
                    manygons.append(poly)
                elif len(poly) == 3:
                    threegons.append(poly)
                    manygons.append(poly)
                else:
                    manygons.append(poly)

            #print(f'Onegons: {onegons}')
            #print(f'Twogons: {twogons}')
            #print(f'Threegons: {threegons}')
            #print(f'Manygons: {manygons}')

            chooser = random.randint(1, 4)
            # print(f'chooser = {chooser}')
            if chooser == 1:
                if len(onegons) == 0:
                    return False
                else:
                    onegon = random.choice(onegons)
                    if onegon[0].p[1] == '*':
                        # print('Failed because trying to unreide1 a circle!')
                        return False
                    # print(f'Reide1 with {onegon}')
                    self.unreide1(*onegon)
                    return True
            elif chooser == 2:
                if len(twogons) == 0:
                    return False
                else:
                    twogon = random.choice(twogons)
                    crossing1 = twogon[0].p[0]
                    crossing2 = twogon[1].p[0]
                    if self.crossings[crossing1] ^ self.crossings[crossing2]:
                        # print(f'Unreide2 with {twogon}')
                        self.unreide2(*twogon)
                        return True
                    else:
                        return False
            elif chooser == 3:
                if len(manygons) == 0:
                    return False
                else:
                    manygon = random.choice(manygons)
                    # a, b = random.sample(manygon, 2)
                    # print(f'Reide2 with edges {[a,b]}')
                    self.reide2(*random.sample(manygon, 2))
                    return True
            else:
                if len(threegons) == 0:
                    return False
                else:
                    threegon = random.choice(threegons)
                    crossing1 = threegon[0].p[0]
                    crossing2 = threegon[1].p[0]
                    crossing3 = threegon[2].p[0]
                    if self.crossings[crossing1] and self.crossings[crossing2] and self.crossings[crossing3]:
                        return False
                    elif not self.crossings[crossing1] and not self.crossings[crossing2] and not self.crossings[crossing3]:
                        return False
                    else:
                        # print(f'Reide3 with {threegon}')
                        self.reide3(*threegon)
                        return True


def balance(diag):
    wrythe = diag.wrythe
    sign = wrythe > 0
    node = random.choice(list(diag.index))
    for _ in range(abs(wrythe)):
        diag.reide1(Edge(diag, node), not sign)

def produce_segments(diag, prefix='a'):
    for root in diag.index:
        if root[1] == '-':
            break
    d = {}
    segdict = {}

    cursor = root
    counter = 1
    while True:
        segment_name = prefix + str(counter)
        counter += 1

        segment_list = [cursor]
        segdict[(cursor[0], 2)] = segment_name
        cursor = diag.forward(cursor)
        while cursor[1] == '+':
            segdict[(cursor[0], 0)] = segment_name
            segment_list.append(cursor)
            cursor = diag.forward(cursor)
        d[segment_name] = segment_list
        segdict[(cursor[0], 1)] = segment_name
        if cursor == root:
            break

    return d, segdict


def get_relators(diag, d, segdict):
    for i, v in diag.crossings.items():
        s = 1 if v else -1
        yield [
            (segdict[(i, 0)], -s),
            (segdict[(i, 1)], 1),
            (segdict[(i, 0)], s),
            (segdict[(i, 2)], -1)
        ]


def get_target_word(diag, d, segdict):
    for root in diag.index:
        break

    word = []

    def inner(root):
        cursor = root
        while True:
            if cursor[1] == '-':
                sign = 1 if diag.crossings[cursor[0]] else -1
                yield segdict[(cursor[0], 0)], sign

            cursor = diag.forward(cursor)
            if cursor == root:
                break

    for a, b in inner(root):
        if len(word) == 0:
            word.append((a, b))
        elif word[-1][0] == a:
            if word[-1][1] + b == 0:
                word.pop()
            else:
                word[-1] = (word[-1][0], word[-1][1] + b)
                # word[-1][1] += b
        else:
            word.append((a, b))

    return word

def double(diag, oldname, newname, entangle=True):
    oldlink = diag.links[oldname]
    corr = {}
    crossings = {c for (c, _) in oldlink}

    newlink = bidict()
    diag.links[newname] = newlink

    for bb in crossings:
        rr = diag.gen_crossing_num()
        corr[bb] = rr
        pos_flag = diag.crossings[bb]
        diag.crossings[rr] = pos_flag
        # diag.index[(rr, '+')] = newname
        # diag.index[(rr, '-')] = newname
        diag.index[(rr, '+')] = newlink
        diag.index[(rr, '-')] = newlink
    
    for basept in oldlink:
        break
    cursor = basept
    alpha = (corr[cursor[0]], cursor[1])

    while True:
        cursor = oldlink[cursor]
        beta = (corr[cursor[0]], cursor[1])

        newlink[alpha] = beta
        alpha = beta
        
        if cursor == basept:
            break

    if entangle:
        for bb in crossings:
            pos_flag = diag.crossings[bb]
            rr = corr[bb]

            # First do BR
            diag.add_crossing(
                Edge(diag, (bb, '+'), forward=pos_flag),
                Edge(diag, (rr, '-'), forward=pos_flag),
                pos_flag
            )
            # Now RB
            diag.add_crossing(
                Edge(diag, (rr, '+'), forward=not pos_flag),
                Edge(diag, (bb, '-'), forward=not pos_flag),
                pos_flag
            )

def apply_random_moves_and_balance(diag, n):
    for i in range(n):
        diag.random_move()
    balance(diag)

def make_cycle(g):
    d = bidict()
    try:
        root = next(g)
        old = root
        for val in g:
            d[old] = val
            old = val
        d[old] = root
    except StopIteration:
        pass
    return d

def polywalk(diag, node, flag_fwd):
    cursor = node
    e = diag.Edge(p=node, forward=flag_fwd)
    yield e.data

    while True:
        # (1) move
        cursor = e.q
        if cursor[0] == node[0]:
            break

        # (2) jump
        if cursor[1] != '*':
            flag_fwd ^= diag.crossings[cursor[0]] ^ (cursor[1] == '+')
        cursor = diag.node_flip(cursor)
        e = Edge(diag, p=cursor, forward=flag_fwd)
        yield e.data
        
# def short_sequence(diag, root_node):
#     link = diag.index[root_node]
#     prev_node = diag.backward[root_node]
#     next_node = diag.forward[root_node]

# def get_bridge(node):
#     pass


# def first_stab(diag, root_node):
#     link = diag.index[root_node]
#     formatter = '{} {} {} {} {}'

#     def h_chunk(cnum, sign):
#         b = sign == '-'
#         return f"{DASH_LEFT if b else '-'}{cnum}{DASH_RIGHT if b else '-'}"


# def zeroth_stab(diag, root_node):
#     link = diag.index[root_node]

#     space_needed = len(str(diag.cursor))
#     fmt = f'{space_needed}d'

#     def wormsign(node):
#         return node_print(node, fmt)

#     for node in iterate_until_repeat(root_node, diag.forward):
#         print(wormsign(node))


def iterate_until_repeat(start, nxtfn, eq_test=None):
    root = start
    cursor = nxtfn(root)

    if eq_test is None:
        eq_test = lambda x, y: x == y

    yield root
    while not eq_test(root, cursor):
        yield cursor
        cursor = nxtfn(cursor)
    

def print_word(word):
    return '*'.join(f'{a}^{b}' for a, b in word)

def do_all():
    do_everything(int(argv[1]))

def do_everything(n):
    circle = LinkDiag(TREFOIL)
    apply_random_moves_and_balance(circle, n)
    d, segdict = produce_segments(circle, 'H.')
    print('### First, the relators:')
    print(', '.join(print_word(w) for w in get_relators(circle, d, segdict)))
    print()
    print('### Now, the target word:')
    print(print_word(get_target_word(circle, d, segdict)))
    print()
    print('### Number of generators:')
    print(len(circle.crossings))
    
def foo(ltr, n):
    elts = [f'{ltr}^{i}' for i in range(-n, n+1)]
    return f'[{",".join(elts)}]'

GAP_MAGIC = "rels:=Union(List(RelatorsOfFpGroup(G),r->List({}, e->r^e)))"

def foo2(ltr, n):
    return GAP_MAGIC.format(foo(ltr, n))



EX_YAML = '''
foo:
    links:
        a: [1+, 2-, 3+, 1-, 4-, 5+, 2+, 3-]
        b: [4+, 5-]
    positive: [1, 2, 3, 4, 5]
'''
FIGURE8 = '''
foo:
    links:
        b: [1+, 1-]
    positive: [1]
'''

KNOTTY = '''
foo:
    links:
        b: [1+, 4-, 3+, 1-, 2+, 3-, 4+, 2-]
    positive: [3, 4]
'''

TREFOIL = '''
foo:
    links:
        b: [1+, 2-, 3+, 1-, 2+, 3-]
    positive: [1,2,3]
'''

CIRCLE = '''
foo:
    links:
        b: [0*]
    positive: []
'''

# Quick comment

if __name__ == '__main__':
    do_everything()