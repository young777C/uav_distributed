# 加强语言 grounding 方案

> 2026-08-11 · 触发:全量 with-language mis_follow 卡在位置 floor 41.6%(ep49=42.4%,tid 训练 loss 降但 val 不降)→ 模型没学会用语言消歧。

## 0. 问题定位(为什么模型没用语言)

**当前 target_id 机制(`stage2.py::_ctx_vec` + `dit.py::TargetIDHead`)**:
```
ctx_vec = mean(vlm_ctx 的 100 token)            # (B,2560) 全局标量摘要
score_i = Linear(GELU( Linear(cand_i) ⊙ Linear(ctx_vec) ))   # 逐元素乘 → 打分
```
三个结构性弱点:
1. **`_ctx_vec` 用 100 token 的均值池化 → 语言被稀释**。语言是融进"被描述那辆车所在位置"的少数 token 里的;mean 把"yellow Jeep 在这里"这个局部信号摊平成全局平均,**锐利的语言 grounding 丢了**。← 最可能的主因
2. **所有候选共用同一个 `ctx_vec`** → 头无法**逐候选**问"这辆车匹配'yellow'吗";只能拿每辆车和同一个模糊摘要比。
3. **`cand ⊙ ctx` 逐元素乘 = 弱交互**,不是显式的"匹配/相似度"。

**外加两个非头因素**:
- **位置泄漏**:`cand_feats`(VLM grid 在车框处 pool)含 ViT 位置编码 → 头能用位置;**位置 floor 41.6% 就是天花板**,模型坐在那说明走了位置捷径。
- **可能:缓存里语言信号本就弱**(待诊断 D2)——若 VLM 没把语言充分染进图像 token,任何头都救不回来。

---

## 1. 先做诊断(便宜,定位根因是"头 / 缓存 / 数据"哪一层)

| 诊断 | 做法 | 判读 |
|---|---|---|
| **D1 w/o-language 消融** | 通用 prompt 重训(进行中) | 无语言≈42% → 模型没用语言;明显更高 → 语言有用只是没压过位置 |
| **D2 缓存语言敏感度** ★ | 同一帧,比 with-lang vs no-lang 缓存的 token 差异 | ✅ **已做(2026-08-11)**:cosine=0.47(中位0.19)、相对L2=2.31 → **语言强烈进了缓存 → 问题在头,不在 backbone**。**修复可在头层面、复用现成缓存、无需重算预计算** |
| **D3 显式语言探针** | 给头额外喂 target 颜色/车型的 one-hot/embedding,只训 target_id 头 | 显式语言能把 mis_follow 压下去 → 信号可用、是**头没提取**;仍压不下 → 数据位置天花板/缓存问题 |

> **D2/D3 是关键**:先花半天把"问题在头 / 缓存 / 数据"分清,再对症下药,别盲目改头。D2 现在就能做(with-lang 缓存已在,no-lang 缓存马上好)。

---

## 2. 改法(按预期 impact 排序)

### #1 ★ cross-attention 头:候选 → 上下文(修稀释 + 逐候选匹配)
**核心改法。** 不再把 ctx 池化成一个向量;让**每个候选作为 query 去 attend 全部 100 个上下文 token**,得到**候选专属、语言感知**的表征再打分:
```python
class TargetIDHead(nn.Module):
    def __init__(self, cand_dim, ctx_dim, d=256, n_heads=4):
        self.q = nn.Linear(cand_dim, d)
        self.kv = nn.Linear(ctx_dim, 2*d)
        self.attn = MultiheadAttention(d, n_heads)   # cand(query) × ctx(key/value)
        self.score = nn.Sequential(nn.LayerNorm(d), nn.GELU(), nn.Linear(d,1))
    def forward(self, cand_feats, vlm_ctx, ctx_mask, cand_mask=None):
        q = self.q(cand_feats)                        # (B,N,d)
        k,v = self.kv(vlm_ctx).chunk(2,-1)            # (B,100,d)
        a = self.attn(q, k, v, key_padding_mask=~ctx_mask)  # 每候选查语言
        return self.score(a).squeeze(-1)              # (B,N)
```
- 改动:`stage2.py` 传 `b["vlm_ctx"]` + `ctx_mask`(而非 `_ctx_vec`);头签名改；两处调用(train+eval)。
- 为什么有效:直接解决弱点 1+2——每辆车能"查"语言里是否描述了自己,不再被均值稀释。

### #2 ★ 提高 `loss.target_id` 权重(最便宜,先试)
`0.2 → 0.5 或 1.0`。当前动作损失权重 1.0、消歧才 0.2,梯度上消歧被压。纯 config 改,零风险,和 #1 可叠加。

### #3 对比 / margin 损失(强化语言→目标信号)
在 cross-entropy 外加:让目标候选的分数比干扰车高出 margin(hinge)或 InfoNCE。把"语言选对目标"变成更强的对比目标,抑制走捷径。

### #4 更强的相似度头
把 `⊙` 换成**学习的相似度**:query/key 归一化后点积(cosine),或双线性 `q^T W k`。比逐元素乘更接近"匹配"语义。可与 #1 合并(attention 本身就是点积匹配)。

### #5 显式语言条件(若 D2/D3 显示 VLM 融合太弱)
不只靠 VLM 隐式融合,把 target 的**颜色/车型词**单独编码(小文本编码器 / 属性 embedding)喂给头。**代价**:偏离"VLM 端到端 grounding"的设计,更像 NL-tracking 的显式匹配;作为兜底 / 诊断,不作首选。

### #6 数据侧:去深度捷径(真天花板,已反馈)
只要位置 floor 41.6%,模型就有捷径可走。`data-fix-spec.md` §4.1 已反馈"随机化目标跟踪距离/深度";floor 逼近随机(~68%)后,模型才**被迫**用语言。**模型改法(#1-#4)让模型会用语言,数据改法让它不得不用**——两者互补,缺一不可。

---

## 3. 建议执行顺序

```
1. D1(消融结果,马上出)+ D2(缓存语言敏感度,现在可跑)+ D3(显式语言探针)
      → 分清问题在 头 / 缓存 / 数据
2. 若问题在头(D2 差异大):#2(提权重,先跑基线)→ #1(cross-attn 头)→ 重训重测
3. 若问题在缓存(D2 差异小):回 backbone —— 检查 prompt 是否被截断/精度/是否 text-before-image 生效
4. 若数据天花板(#1+#2 后仍卡 ~41%):回数据侧 #6 去深度捷径,与模型改法并行
```

**最可能的组合**:#2(提权重)+ #1(cross-attn 头)解决"头稀释语言" + 数据侧 #6 去深度捷径抬天花板。三者一起,才可能把 mis_follow 从 ~42% 压到真正体现语言价值的水平。
