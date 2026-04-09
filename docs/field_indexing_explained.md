# mFAR Field Indexing: What Gets Indexed?

## TL;DR

**具体实体名称（VCP, TRAF6等）都被索引了。** 类型标签（gene/protein, disease等）作为前缀也在索引里，不会替代实体名。

---

## 你的例子

原始corpus数据：
```json
{
  "name": "lipid droplet",
  "type": "cellular_component",
  "interacts with": {
    "gene/protein": ["VCP", "TRAF6"]
  }
}
```

`interacts with` field建索引时，先经过 `format_dict()` 转成纯文本：

```
gene/protein: VCP, TRAF6
```

然后分别建两种索引：

- **BM25 sparse索引**（`interacts with_sparse`）：对这个文本做tokenize → 倒排索引。搜"VCP"能命中，搜"TRAF6"也能命中，搜"gene/protein"也能命中。
- **Dense索引**（`interacts with_dense`）：用contriever encoder把这个文本编码成768维向量。搜"proteins that interact with VCP"通过向量相似度命中。

**结论：VCP和TRAF6都在索引里。**

---

## 格式化逻辑（`mfar/data/format.py`）

### 核心函数：`format_dict()`

对于PRIME dataset的relation fields，原始数据结构是：
```json
{field_name: {sub_type: [entity1, entity2, ...]}}
```

`format_dict()` 处理逻辑：

```python
for key in item_dict:          # key = "gene/protein"
    value = item_dict[key]     # value = ["VCP", "TRAF6"]
    
    if isinstance(value, list):
        items = ", ".join(value)                # "VCP, TRAF6"
        all_strings.append(f"{key}: {items}")   # "gene/protein: VCP, TRAF6"

return "; ".join(all_strings)
```

**关键：`", ".join(value)` 把列表里每个实体名都拼进去了。**

### 多子类型的情况

很多field有多个子类型。例如 `interacts with`：

```json
"interacts with": {
  "cellular_component": ["spliceosomal complex", "nucleoplasm"],
  "molecular_function": ["metal ion binding", "protein binding"],
  "pathway": ["mRNA Splicing - Minor Pathway"],
  "biological_process": ["RNA splicing"]
}
```

格式化结果：
```
cellular_component: spliceosomal complex, nucleoplasm; molecular_function: metal ion binding, protein binding; pathway: mRNA Splicing - Minor Pathway; biological_process: RNA splicing
```

每个子类型的所有实体名都保留了，用`"; "`分隔不同子类型。

---

## 普遍情况：各field的格式化pattern

### 1. 简单dict → list（最常见）

大部分relation fields都是这个结构。

| Field | 原始数据 | 索引文本 |
|-------|---------|---------|
| `ppi` | `{"gene/protein": ["TTR", "NFE2", "DYRK1A"]}` | `gene/protein: TTR, NFE2, DYRK1A` |
| `associated with` | `{"disease": ["skin carcinoma", "Bowen disease"]}` | `disease: skin carcinoma, Bowen disease` |
| `expression present` | `{"anatomy": ["pituitary gland", "lymph node"]}` | `anatomy: pituitary gland, lymph node` |
| `expression absent` | `{"anatomy": ["quadriceps femoris", "thymus"]}` | `anatomy: quadriceps femoris, thymus` |
| `indication` | `{"disease": ["diabetes", "hypertension"]}` | `disease: diabetes, hypertension` |
| `contraindication` | `{"disease": ["renal failure"]}` | `disease: renal failure` |
| `side effect` | `{"effect/phenotype": ["nausea", "headache"]}` | `effect/phenotype: nausea, headache` |
| `target` | `{"gene/protein": ["CYP3A4", "ABCB1"]}` | `gene/protein: CYP3A4, ABCB1` |
| `carrier` | `{"gene/protein": ["ALB"]}` | `gene/protein: ALB` |
| `enzyme` | `{"gene/protein": ["CYP2D6"]}` | `gene/protein: CYP2D6` |

### 2. 多子类型dict

`interacts with` 和 `parent-child` 经常有多个子类型：

| Field | 原始数据 | 索引文本 |
|-------|---------|---------|
| `interacts with` | `{"cellular_component": ["nucleus"], "pathway": ["apoptosis"]}` | `cellular_component: nucleus; pathway: apoptosis` |
| `parent-child` | `{"disease": ["cancer"], "anatomy": ["lung"]}` | `disease: cancer; anatomy: lung` |

### 3. 非dict类型（details等）

`details` field是嵌套dict，直接递归格式化：

```json
"details": {
  "Description": "Ivermectin is a broad-spectrum anti-parasite medication...",
  "Half Life": "16 hours",
  "Mechanism": "Binds to glutamate-gated chloride channels..."
}
```

索引文本：
```
Description: Ivermectin is a broad-spectrum anti-parasite medication...; Half Life: 16 hours; Mechanism: Binds to glutamate-gated chloride channels...
```

### 4. Single field（全文档）

除了per-field索引，还有 `single_dense` 和 `single_sparse` 索引，把整个文档格式化成一个长文本（通过 `format_prime()`），包含name、type、details和所有relations。

---

## 数据流总图

```
corpus文件中的JSON
  │
  ├─ field="ppi" ──→ format_dict({"gene/protein": ["TTR", ...]})
  │                    → "gene/protein: TTR, NFE2, DYRK1A, ..."
  │                    ├─→ BM25 tokenize → sparse索引 (ppi_sparse)
  │                    └─→ contriever encode → dense向量 (ppi_dense)
  │
  ├─ field="indication" ──→ format_dict({"disease": ["diabetes"]})
  │                          → "disease: diabetes"
  │                          ├─→ BM25 → sparse索引 (indication_sparse)
  │                          └─→ contriever → dense向量 (indication_dense)
  │
  ├─ field="interacts with" ──→ format_dict({"gene/protein": ["VCP"], "pathway": [...]})
  │                              → "gene/protein: VCP; pathway: ..."
  │                              ├─→ BM25 → sparse索引
  │                              └─→ contriever → dense向量
  │
  └─ field="single" ──→ format_prime(整个doc)
                         → "- name: PHYHIP\n- type: gene/protein\n- details: ...\n- relations: ..."
                         ├─→ BM25 → sparse索引 (single_sparse)
                         └─→ contriever → dense向量 (single_dense)
```

---

## 验证方法

如果想亲眼看某个文档某个field索引了什么内容：

```python
from mfar.data.format import format_dict
import json

# 读一条文档
with open("data/prime/corpus") as f:
    line = f.readline()
    idx, json_str = line.strip().split("\t", 1)
    doc = json.loads(json_str)

# 看某个field格式化后的内容
field = "interacts with"
if field in doc:
    print(format_dict(doc[field]))
```

---

## 总结

| 问题 | 回答 |
|------|------|
| 索引只有类型（gene/protein）没有实体名（VCP）？ | **不是。** 两者都有，格式是 `"gene/protein: VCP, TRAF6"` |
| BM25能搜到具体实体名吗？ | **能。** "VCP"是索引文本里的token |
| Dense能搜到语义相关的query吗？ | **能。** 整段文本被encoder编码成向量 |
| 类型标签有什么用？ | 帮助区分同名实体在不同子类型下的含义，也作为额外的检索信号 |

---

## Multihop Queries的局限性

索引内容没问题，但mFAR在multihop query上效果差有另一个原因：**mFAR是单跳检索架构**。

### 例子

Query: *"What diseases are associated with proteins that interact with VCP?"*

人的推理是两步：
1. 搜`ppi` field → 找到和VCP交互的蛋白（比如TRAF6, UBXN1）
2. 拿这些蛋白名去搜`associated with` field → 找到相关疾病

### mFAR实际做了什么

mFAR只做一步：把整个query直接和所有doc的所有field做匹配。

```
query = "diseases associated with proteins that interact with VCP"
    ↓
同时在46个索引上搜：
  - ppi_dense:     cos_sim(query_emb, doc_ppi_emb)      → 能匹配到含VCP的ppi field
  - associated_with_dense: cos_sim(query_emb, doc_assoc_emb) → 能匹配到含disease的field
  - ...
    ↓
softmax加权合并 → 排序
```

**问题**：mFAR可能给一个ppi field里有"VCP"的doc打高分（第一跳匹配），但这个doc的`associated with` field可能和query问的疾病无关。它不会把第一跳的结果（TRAF6）传给第二跳去搜。

### 为什么索引不是问题

"VCP"确实在索引里——BM25搜"VCP"能命中ppi field里有VCP的doc。第一跳的匹配没问题。

问题在于**没有第二跳**。mFAR找到了"和VCP交互的蛋白"所在的doc，但它不会用这个中间结果去进一步检索"这些蛋白关联的疾病"。它把两跳压缩成了一跳，依赖query embedding和field embedding的直接相似度来隐式完成跨field推理——这对复杂的multihop query不够用。

### 可能的解决方向

1. **Query Decomposition**: 把multihop query拆成多个子query，分步检索
2. **Graph Traversal**: 利用KG结构做多跳遍历，而不是纯向量检索
3. **Retrieval-Augmented Reasoning**: 先检索第一跳结果，用LLM生成第二跳query，再检索
