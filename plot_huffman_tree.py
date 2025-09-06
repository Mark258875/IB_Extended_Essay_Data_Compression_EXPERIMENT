# Fix the layout assignment: use post-order so both children have positions before computing internal node x
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

# Rebuild tree structures fresh (to avoid state from previous run)
class Node:
    __slots__ = ("left","right","label","code","prob")
    def __init__(self):
        self.left = None
        self.right = None
        self.label = None
        self.code = None
        self.prob = None

codes = [
    ("S1","0",        "1/2"),
    ("S2","10",       "1/4"),
    ("S3","110",      "1/8"),
    ("S4","1110",     "1/16"),
    ("S5","111100",   "1/64"),
    ("S6","111101",   "1/64"),
    ("S7","111110",   "1/64"),
    ("S8","111111",   "1/64"),
]

root = Node()
for label, code, prob in codes:
    cur = root
    for i,bit in enumerate(code):
        if bit == "0":
            if cur.left is None:
                cur.left = Node()
            if i == len(code)-1:
                cur.left.label = label
                cur.left.code = code
                cur.left.prob = prob
            else:
                cur = cur.left
        else:
            if cur.right is None:
                cur.right = Node()
            if i == len(code)-1:
                cur.right.label = label
                cur.right.code = code
                cur.right.prob = prob
            else:
                cur = cur.right

positions = {}
levels = {}
leaf_x = 0

def assign_positions(node, depth=0):
    global leaf_x
    if node is None:
        return
    assign_positions(node.left, depth+1)
    assign_positions(node.right, depth+1)
    if node.left is None and node.right is None:
        x = leaf_x
        leaf_x += 1
    else:
        if node.left is not None and node.right is not None:
            x = (positions[node.left][0] + positions[node.right][0]) / 2
        elif node.left is not None:
            x = positions[node.left][0]
        else:
            x = positions[node.right][0]
    positions[node] = (x, depth)
    levels[node] = depth

assign_positions(root, 0)

# Collect edges
edges = []
def collect_edges(node):
    if node is None: return
    if node.left is not None:
        edges.append((node, node.left, "0"))
        collect_edges(node.left)
    if node.right is not None:
        edges.append((node, node.right, "1"))
        collect_edges(node.right)
collect_edges(root)

# Plot
fig, ax = plt.subplots(figsize=(7, 8))

# Draw edges (dark blue)
for parent, child, label in edges:
    x1, y1 = positions[parent]
    x2, y2 = positions[child]
    y1_draw = -y1
    y2_draw = -y2
    ax.add_line(Line2D([x1, x2], [y1_draw, y2_draw], color='darkblue'))
    mx, my = (x1 + x2)/2, (y1_draw + y2_draw)/2
    ax.text(mx, my + 0.1, label, fontsize=9, ha='center', va='bottom', color='darkblue')

# Draw nodes (red) and annotate leaves
def draw_node(node):
    if node is None: return
    x, y = positions[node]
    y = -y
    circ = Circle((x, y), 0.12, fill=True, facecolor='navy', edgecolor='black')
    ax.add_patch(circ)
    if node.left is None and node.right is None:
        # Put code to the right for clarity
        ax.text(x + 0.18, y, f"{node.code}\n(p={node.prob})", fontsize=9, ha='left', va='center')
    draw_node(node.left)
    draw_node(node.right)

draw_node(root)

ax.set_title("Huffman Tree (vertical; left=0, right=1)")
ax.set_aspect('equal')
ax.axis('off')

xs = [pos[0] for pos in positions.values()]
ys = [-pos[1] for pos in positions.values()]
ax.set_xlim(min(xs)-0.8, max(xs)+0.8)
ax.set_ylim(min(ys)-0.8, max(ys)+0.5)

plt.tight_layout()
plt.show()
