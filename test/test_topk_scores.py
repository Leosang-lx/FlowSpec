from eagle.ea_model import EaModel
from transformers import AutoConfig
import torch
import os
import json
from config.run_config import config as run_config
from stage_ea_model import StageEaModel
from config.run_config import config
import numpy as np


cache_dir = '/home/liux/big_file/'
base_model_path = 'meta-llama/Llama-2-7b-chat-hf'
EAGLE_model_path = 'yuhuili/EAGLE-llama2-chat-7B'
base_model_path = cache_dir + base_model_path
EAGLE_model_path = cache_dir + EAGLE_model_path

config = AutoConfig.from_pretrained(base_model_path)

# print(config)

model = EaModel.from_pretrained(
    base_model_path=base_model_path,
    ea_model_path=EAGLE_model_path,
    torch_dtype=torch.float16,
    # low_cpu_mem_usage=True,
    device_map="cuda:0",
    total_token=13,
    depth=3
)

print('EaMoel loaded')
model.eval()
tokenizer = model.tokenizer


prompt = "Hello, how are you?"
input_ids = tokenizer([prompt]).input_ids
input_ids = torch.as_tensor(input_ids).cuda()

def is_valid(selected_indices, parents_list2):
    # the parents of the 2nd half tree are all selected, thus connected
    selected_indices = set(selected_indices)
    half2_parents = set(parents_list2)
    diff = half2_parents - selected_indices
    assert not diff, f'The set of the 2nd half tree parents is not a subset of the two half trees, diff: {diff}'

def test_expand(input_ids):
    outputs, orig, hidden_states = model(input_ids, past_key_values=None, output_orig=True)
    token = torch.argmax(orig[:, -1])[None, None]
    input_ids_ea = torch.cat((input_ids, token), dim=1)

    draft_tokens, retrieve_indices, tree_mask, tree_position_ids, last_state = model.ea_layer.topK_genrate(
        hidden_states,
        input_ids_ea,
        model.base_model.lm_head,
        None,
        total_tokens=20,
        depth=3,
        log=True,
        return_last=True,
        sort_score=True,
    )
    print(f'========Draft tokens========')
    print(f'draft_tokens: {draft_tokens}')
    print(f'retrieve_indices: {retrieve_indices}')
    # print(f'tree_mask: {tree_mask}')
    print(f'tree_position_ids: {tree_position_ids}')
    print(f'retrieve_position_ids: {tree_position_ids[retrieve_indices]}')

    # topk_cs_index, scores, ss_token, scores_list, parents_list, last_top_scores_index = last_state
    last_depth,\
    last_input_ids, last_input_hidden, past_key_values,\
    draft_tree_mask, len_posi, top_k,\
    topk_cs_index, scores, ss_token, scores_list, parents_list, last_top_scores_index = last_state
    last_tree = draft_tokens, retrieve_indices, tree_mask, tree_position_ids

    # ========= Convert cumulative log-probs to cumulative probabilities (product of probs) =========
    # last_state[10]: scores_list (list of GPU tensors, unflattened)
    # last_state[12]: last_top_scores_index (numpy array, indices into flattened scores_list)
    scores_list_raw = last_state[10]  # list of GPU tensors
    flat_scores = torch.cat(scores_list_raw, dim=0).view(-1).cpu().numpy()
    top_idx = last_state[12]
    # cumulative probability = exp(sum of log_probs) = product of probabilities along path
    # cumulative_probs = np.exp(flat_scores[top_idx])
    cumulative_probs = flat_scores[top_idx]

    total_draft_tokens = len(cumulative_probs)
    print(f'\n========Cumulative probabilities (product of probabilities along path)========')
    for i in range(total_draft_tokens):
        tid = draft_tokens[0, i + 1].item()
        print(f'Draft token [{i+1:2d}] (id={tid:6d}): cumulative prob = {cumulative_probs[i]:.8f}')

    # ========= Build score map for all draft token positions =========
    # score_map[pos] maps draft_tokens[0, pos] -> cumulative probability
    score_map = np.zeros(draft_tokens.shape[1])  # (total_tokens+1,)
    score_map[0] = 0.0  # root: log(1) = 0
    for i in range(1, draft_tokens.shape[1]):
        score_map[i] = cumulative_probs[i - 1]

    # ========= Build tree structure from retrieve_indices =========
    retrieve_np = retrieve_indices.cpu().numpy() if torch.is_tensor(retrieve_indices) else retrieve_indices

    # Build children dict: children[parent_pos] = [child_pos1, ...], sorted by score descending
    children = {}
    for path in retrieve_np:
        valid = path[path >= 0]
        for i in range(len(valid) - 1):
            p, c = int(valid[i]), int(valid[i + 1])
            children.setdefault(p, [])
            if c not in children[p]:
                children[p].append(c)
    for p in children:
        children[p] = sorted(children[p], key=lambda c: score_map[c], reverse=True)

    # ========= Tree layout (x, y positions, root at top) =========
    x_pos = {}
    def assign_x(node, offset=0):
        if node not in children or not children[node]:
            x_pos[node] = offset
            return offset + 1
        child_xs = []
        for child in children[node]:
            offset = assign_x(child, offset)
            child_xs.append(x_pos[child])
        x_pos[node] = (child_xs[0] + child_xs[-1]) / 2
        return offset
    total_leaves = assign_x(0)

    y_pos = {}
    def assign_y(node, depth=0):
        y_pos[node] = depth
        if node in children:
            for child in children[node]:
                assign_y(child, depth + 0.8)
    assign_y(0)
    max_depth = max(y_pos.values()) + 1

    # ========= Colormap setup =========
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    draft_scores = score_map[1:]  # exclude root
    vmin, vmax = draft_scores.min(), score_map.max()
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.viridis

    # ========= Text tree output =========
    print(f'\n========Tree structure with scores and colors========')
    print(f'Colormap: viridis | Normalize: vmin={vmin:.2f}, vmax={vmax:.2f}')
    def print_tree(node, line_prefix='', children_prefix=''):
        label = 'Root' if node == 0 else f'[{node}]'
        tid = draft_tokens[0, node].item()
        score = score_map[node]
        rgba = cmap(norm(score))
        rgb = tuple(int(round(c * 255)) for c in rgba[:3])
        print(f'{line_prefix}{label} token_id={tid:6d}  score={score:+.2f}  RGB{rgb}')
        if node in children:
            for i, child in enumerate(children[node]):
                is_last = i == len(children[node]) - 1
                conn = '└── ' if is_last else '├── '
                nxt = '    ' if is_last else '│   '
                print_tree(child, children_prefix + conn, children_prefix + nxt)

    print_tree(0)

    # ========= Tree-structured heatmap image =========
    fig, ax = plt.subplots(figsize=(max(6, total_leaves * 0.8), max(4, max_depth * 1.5)))

    # Draw edges
    for p in children:
        for c in children[p]:
            ax.plot([x_pos[p], x_pos[c]], [y_pos[p], y_pos[c]],
                    color='gray', linewidth=1.0, zorder=1)

    # Draw nodes
    node_list = list(range(draft_tokens.shape[1]))
    xs = [x_pos[n] for n in node_list]
    ys = [y_pos[n] for n in node_list]
    node_scores = [score_map[n] for n in node_list]

    # 每个节点的轮廓颜色, 按 draft_tokens 位置 [0=root, 1..20=草稿]
    # 格式: RGB tuple, 0-1 范围或 0-255; 或颜色名如 'red', 'gold', 'black'
    # node_edge_colors = ['black'] * draft_tokens.shape[1]
    node_edge_colors = ['red'] * 5 + ['black'] * (draft_tokens.shape[1] - 5)
    # node_edge_colors = ['#FFD700'] + ['black'] * (draft_tokens.shape[1] - 1)  # 金色根节点

    scatter = ax.scatter(xs, ys, s=600, c=node_scores, cmap=cmap, norm=norm,
                         edgecolors=node_edge_colors, linewidths=1.5, zorder=2)

    # Annotate nodes with score values
    for n in node_list:
        score = score_map[n]
        tc = 'white' if norm(score) < 0.5 else 'black'
        ax.text(x_pos[n], y_pos[n], f'{score:.2f}',
                ha='center', va='center', fontsize=9, fontweight='bold',
                color=tc, zorder=3)

    ax.set_aspect('equal')
    ax.set_xlim(-0.5, total_leaves - 0.5)
    ax.set_ylim(max_depth - 0.3, -0.3)

    # Align colorbar with tree extent, not full axes padding
    tree_span = max(y_pos.values()) - min(y_pos.values())
    axes_span = abs(ax.get_ylim()[0] - ax.get_ylim()[1])
    shrink = tree_span / axes_span / 2

    cbar = plt.colorbar(scatter, ax=ax, shrink=shrink)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('Cumulative log-prob', fontsize=8)
    ax.axis('off')
    # ax.set_title('Draft Token Tree (color = cumulative log-prob)', fontsize=10, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), 'topk_tree.pdf')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'\nTree image saved to: {output_path}')

    plt.close(fig)

    return draft_tokens, retrieve_indices, cumulative_probs


test_expand(input_ids)