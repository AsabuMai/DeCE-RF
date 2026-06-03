# RF Local Editing 当前问题归因与 V2 改进记录

## 0. 当前结论概览

当前方法的主要优势不是“编辑更强”，而是“背景保持更稳”。这会带来一个风险：如果目标没有真正被编辑出来，较好的 outside L1 / SSIM 可能只是因为模型没有充分修改图像。因此，后续实验不能继续只证明 preservation，而必须证明：

> 在目标编辑确实成功的前提下，方法仍然能保持更好的背景与非编辑区域。

当前核心瓶颈可以概括为：

```text
v_edit ≈ v_target(z_t, t) - v_source(z_t, t)
```

这种 flow velocity 差分在大语义、明显目标上有效，例如 crown、sunglasses、heart。但在小目标、logo、tag、leaf 等任务中，`v_target - v_source` 在局部 support 内没有形成足够明确的 object-forming direction，所以结果往往只是局部纹理、阴影、边缘变化，而不是真正生成目标对象。

因此，接下来不应该继续围绕 `rmsgap` 微调，而应该把方法升级为：

```text
core target transport
+ local flow refresh
+ crop-local editing
+ outside trajectory lock
```

---

## 1. 当前方法的问题

### 1.1 当前编辑主要依赖 velocity residual

当前方法近似为：

```text
v_total = v_source + λ M_edit · (v_target - v_source) + small local CLIP/text gradient
```

其中：

- `v_source`：source prompt 下的 RF velocity；
- `v_target`：target prompt 下的 RF velocity；
- `M_edit`：局部 support / edit mask；
- `λ`：controller 或 edit strength；
- local CLIP/text gradient：辅助的局部语义引导。

这个机制在目标语义明显时可以工作，因为 target 和 source 的 velocity field 差异足够强。例如：

```text
cat_crown
狗戴墨镜 dog_sunglasses
mug_heart
```

但对于小目标或弱局部语义任务，差分方向经常不够明确：

```text
tshirt_star
tote_leaf
cat_replace_bell_heart_tag
rabbit_sunglasses
```

这些 case 中，模型更容易产生：

```text
局部纹理变化
阴影变化
边缘变形
颜色倾向变化
轻微模糊
```

而不是清楚的 star / leaf / tag / sunglasses。

### 1.2 rmsgap 的作用有限

`support_v3_controller_rmsgap` 的真实机制是：

```text
late-stage low-residual finishing boost
```

也就是说，它主要在后期 residual gap 已经较小时增强 edit force，用于 finishing。它不能解决以下问题：

```text
目标对象从一开始就没有形成；
局部 target direction 只是纹理残差；
support 太小导致目标没有生成空间；
小目标在整图语义中权重太低。
```

因此，继续微调 rmsgap 只能放大已有方向。如果已有方向不是 object formation，那么结果仍然不会变成真正的目标对象。

---

## 2. V2 方法方向：Region-Conditioned Target Transport

下一版方法不应只是 scalar controller，而应改为 region-conditioned transport。核心问题从：

```text
已有 edit force 要放大多少？
```

改成：

```text
core 区域应该走 source flow、target flow，还是 target-conditioned local trajectory？
```

建议的新方法名称可以是：

```text
Region-Conditioned Target Transport for RF Local Editing
```

或者：

```text
Core-Target Transport with Outside Trajectory Lock
```

---

## 3. 机制一：Core Target Transport

### 3.1 从 residual edit 改为 target-flow takeover

当前方法偏向：

```text
v_source + λ(v_target - v_source)
```

V2 应该改成：

```text
outside: source / reconstruction flow
ring: weak target-flow blend
core: strong target-flow transport
```

形式上可以写为：

```text
v_total =
    M_out  · v_src
  + M_ring · [(1 - β_r(t)) v_src + β_r(t) v_tar]
  + M_core · [(1 - β_c(t)) v_src + β_c(t) v_tar + γ_c(t)(v_tar - v_src)]
```

其中：

```text
β_c(t): core 区域 target-flow 接管强度
γ_c(t): core 区域 extra edit force / over-transport
β_r(t): ring 区域弱融合强度
```

直觉：

```text
outside 继续走 source，保证背景；
ring 轻微跟 target，保证边界自然；
core 早中期直接靠 target flow，不再只靠微弱差分。
```

### 3.2 初始 schedule

RF sampling 通常从高 t 到低 t。建议使用 early / middle / late 三段式：

```text
t > 0.65:
    β_c = 1.0
    γ_c = 0.5
    β_r = 0.25
    outside_lock = 0.5

0.35 < t <= 0.65:
    β_c = 0.8
    γ_c = 0.25
    β_r = 0.35
    outside_lock = 0.75

t <= 0.35:
    β_c = 0.35
    γ_c = 0.0
    β_r = 0.15
    outside_lock = 0.9
```

解释：

- early / high t：core 区域目标生成最重要，应给 target flow 最大自由度；
- middle：继续形成目标，同时开始加强边界和 preserve；
- late：收敛细节，降低 over-transport，避免背景和边界漂移。

这个机制和 rmsgap 可以共存：

```text
early/middle: target transport 负责生成目标；
late: rmsgap 负责 residual finishing。
```

---

## 4. 机制二：Local Flow Refresh

### 4.1 为什么需要 refresh

小目标失败的一个原因是 core 区域过度贴着 source trajectory。对于 logo、tag、star 这类局部小对象，如果 early stage 已经被 source trajectory 锁定为衣服纹理、布袋纹理或铃铛结构，后续的 `v_target - v_source` 很难把它改成新对象。

因此，需要在 RF path 中给 core 区域更多自由度。这里不是 DDPM 式全图加噪，而是 flow 版本的局部 refresh。

### 4.2 优先使用 deterministic target-prior refresh

建议先做确定性版本，而不是一开始加入 random noise。

利用 clean estimate：

```text
x0_src = z_t - t · v_src
x0_tar = z_t - t · v_tar
```

估计当前 source path 的 residual / noise direction：

```text
eps_hat = (z_t - x0_src) / (t + eps)
```

构造 target-conditioned trajectory point：

```text
z_tar_path_next = x0_tar + t_next · eps_hat
```

然后只在 core 区域 blend：

```text
z_next = (1 - ρ(t)) z_next + ρ(t) z_tar_path_next
```

只作用于 `M_core`。

### 4.3 伪代码

```python
# current step: z_t, t -> t_next

v_src = model(z_t, t, source_prompt)
v_tar = model(z_t, t, target_prompt)

x0_src = z_t - t * v_src
x0_tar = z_t - t * v_tar

eps_hat = (z_t - x0_src) / max(t, eps)
z_tar_path_next = x0_tar + t_next * eps_hat

# normal RF update
z_next = euler_step(z_t, v_total, t, t_next)

rho = refresh_schedule(t)

z_next = (
    M_core * ((1 - rho) * z_next + rho * z_tar_path_next)
    + M_ring * z_next
    + M_out * z_next
)
```

这个操作的含义是：

> 把 core 区域从 source-conditioned RF trajectory 局部切换到 target-conditioned RF trajectory。

它比直接 random perturbation 更可控，也更适合写进论文方法。

### 4.4 small random refresh 作为后续 ablation

如果 deterministic target-prior refresh 不够，可以加入小随机扰动：

```text
z_refresh = z_tar_path_next + σ(t) ε
```

但应满足：

```text
只在 early / middle 阶段；
只作用于 core；
σ 很小；
ring / outside 不加随机扰动；
late 阶段完全关闭。
```

初始参数：

```text
t > 0.65:
    rho = 0.35 ~ 0.55
    sigma = 0.02 ~ 0.05

0.35 < t <= 0.65:
    rho = 0.15 ~ 0.30
    sigma = 0.00 ~ 0.02

t <= 0.35:
    rho = 0
    sigma = 0
```

优先级：

```text
target-prior refresh > random refresh
```

---

## 5. 机制三：Outside Trajectory Lock

一旦 core target transport 变强，outside 区域不能只靠 soft preservation。否则高强度编辑会导致全图漂移。

建议每一步更新后使用 trajectory lock：

```python
z_next = (
    M_out  * z_src_next
    + M_ring * ((1 - alpha_ring) * z_src_next + alpha_ring * z_next)
    + M_core * z_next
)
```

其中：

```text
M_out: 非编辑区域，直接锁 source trajectory；
M_ring: 边界过渡；
M_core: 允许强编辑。
```

初始参数：

```text
alpha_ring early  = 0.6
alpha_ring middle = 0.5
alpha_ring late   = 0.3
```

作用：

```text
允许 core 区域强编辑；
同时避免 outside 区域跟着 target prompt 漂移；
通过 ring blend 减少边界割裂。
```

---

## 6. 机制四：Crop-Local Editing

### 6.1 为什么小目标必须 crop-local

对于以下 case：

```text
tshirt_star
tote_leaf
cat_replace_bell_heart_tag
rabbit_sunglasses
```

全图编辑很吃亏，因为：

```text
目标区域在整图中占比太小；
CLIP/text signal 被整图语义淹没；
flow model 的 target velocity 更关注大结构；
support 内缺乏足够上下文生成新对象。
```

因此，crop-local 是最高优先级的工程改法。

### 6.2 Crop-local pipeline

建议流程：

```text
1. 根据 bbox/support 取局部 crop；
2. crop 扩大 1.5x–2.5x，保留上下文；
3. 将 crop resize 到模型工作分辨率；
4. 在 crop 上运行 local RF edit；
5. 将编辑结果 resize 回原位置；
6. 使用 mask + feather/ring blend 回全图；
7. 可选：全图运行 very weak harmonization。
```

### 6.3 局部 prompt

crop-local 不应继续使用全图 prompt，而应改为局部 prompt。

示例：

```text
tshirt_star
source local prompt:
    close-up of a plain t-shirt chest area

target local prompt:
    close-up of a t-shirt chest area with a clear star logo
```

```text
tote_leaf
source local prompt:
    close-up of a plain tote bag surface

target local prompt:
    close-up of a tote bag surface with a clear green leaf logo
```

```text
cat_replace_bell_heart_tag
source local prompt:
    close-up of a cat collar with a small bell

target local prompt:
    close-up of a cat collar with a visible heart-shaped tag
```

### 6.4 crop-local 的 mask 设计

在 crop 坐标中使用三层 mask：

```text
core:
    目标实际区域；

edit:
    core dilate 15%–30%；

ring:
    edit 再 dilate 15%–30%；

outside crop:
    局部 preserve 区域。
```

小目标不能只用一个很小的 core mask。目标生成需要周围空间。

### 6.5 自动触发规则

可以定义一个简单规则：

```text
if support_area / image_area < 0.03:
    use crop-local editing
else:
    use full-image editing
```

这个规则可以写进论文方法中，不属于单独为某个 case 手调。

---

## 7. Operation-Specific Presets

当前任务不能用一个 controller 统一处理。建议至少分为 add、replace、remove 三类 preset。

### 7.1 Add preset

适用：

```text
cat_crown
dog_sunglasses
mug_heart
tshirt_star
tote_leaf
rabbit_sunglasses
```

目标：生成新的局部对象或图案。

建议机制：

```text
core target transport: medium-high
target-prior refresh: small target 时开启
old-object suppression: off
outside lock: high
crop-local: 小目标时开启
```

初始参数：

```text
β_c early/mid = 1.0 / 0.8
γ_c early/mid = 0.4 / 0.2
rho early/mid = 0.4 / 0.2
outside_lock = 0.75 ~ 0.9
```

### 7.2 Replace preset

适用：

```text
backpack_replace_patch_blue
dog_replace_tennis_ball_star
cat_replace_bell_heart_tag
```

目标：新物体出现，同时旧物体消失。

需要：

```text
target object formation
old object suppression
```

可写为：

```text
v_replace = v_target + η_replace (v_target - v_source) - η_old · v_old_object
```

如果暂时不好实现 negative velocity，可以先使用局部 CLIP/text gradient suppression：

```text
maximize local similarity to target phrase
minimize local similarity to old object phrase
```

示例：

```text
dog_replace_tennis_ball_star
target phrase:
    star-shaped toy
old object phrase:
    tennis ball
```

初始参数：

```text
β_c early/mid = 1.0 / 0.9
γ_c early/mid = 0.6 / 0.3
rho early/mid = 0.5 / 0.25
ring = lower than add, avoid spreading
outside_lock = high
```

### 7.3 Remove preset

适用：

```text
backpack_remove_toy_charm
```

Remove 需要：

```text
old object suppression
background / material fill-in
boundary reconstruction
```

不能只靠：

```text
target prompt = backpack without toy charm
```

建议 prompt：

```text
source:
    backpack with a toy charm hanging from it

target:
    backpack with no toy charm, clean backpack surface, no hanging object

suppression phrase:
    toy charm, hanging charm, keychain, dangling object
```

schedule：

```text
early:
    suppress object

middle:
    move toward no-object target

late:
    reconstruct local texture and boundary
```

---

## 8. 评价方式必须调整

### 8.1 避免“没改所以保持好”

不能只报告：

```text
outside L1 更低
outside SSIM 更高
```

因为这可能只是 under-edit。

正确评价应该是：

```text
在 edit success 成立的样本中，比较 preservation；
或者在相同 preservation budget 下，比较 edit success / edit strength。
```

### 8.2 引入 Edit Success Threshold

每个 case 应定义 edit success：

```text
tshirt_star:
    star visible = success

tote_leaf:
    leaf logo visible = success

red_chair_blue:
    chair region blue ratio above threshold = success

backpack_remove_toy_charm:
    charm residual below threshold = success

dog_replace_tennis_ball_star:
    star-like object visible and tennis ball suppressed = success
```

### 8.3 推荐指标

Edit strength / success：

```text
local CLIP(target phrase) ↑
VLM yes/no target exists ↑
color ratio / color distance for recolor
old-object residual score ↓ for remove / replace
```

Preservation：

```text
outside L1 ↓
outside SSIM ↑
outside LPIPS ↓
```

最终主结果应是二维 trade-off：

```text
edit strength ↑ vs preservation damage ↓
```

### 8.4 Matched-edit comparison

不要只比较：

```text
fixed@scale1.0 vs ours@scale1.0
```

应该比较：

```text
fixed 调到达到 edit success threshold；
ours 调到达到相同 edit success threshold；
然后比较 outside L1 / SSIM。
```

也就是：

```text
same edit success, better preservation
```

或者：

```text
same preservation budget, better edit success
```

---

## 9. Case 筛选与实验集合

### 9.1 当前 12 cases 的分层

主 sweep 推荐 7 个：

```text
cat_crown
dog_sunglasses
mug_heart
red_chair_blue
backpack_remove_toy_charm
backpack_replace_patch_blue
dog_replace_tennis_ball_star
```

作用：

```text
cat_crown / dog_sunglasses / mug_heart:
    easy / non-regression anchor；证明强方法不会破坏已成功编辑。

red_chair_blue / backpack_remove_toy_charm / backpack_replace_patch_blue / dog_replace_tennis_ball_star:
    stress / trade-off cases；真正用于证明 Pareto frontier。
```

暂时不适合直接进入主 sweep 的 case：

```text
tshirt_star:
    star 没真正出现，只改了衣服边缘/阴影。

tote_leaf:
    leaf logo 没出来。

cat_replace_bell_heart_tag:
    目标太小，几乎看不到有效替换。

rabbit_sunglasses:
    没加出 sunglasses，还改了兔子身体纹理。

dog_crown:
    crown 被图像上边裁掉，paper evidence 风险大。
```

这些 case 不应该直接丢弃，而应作为 V2 edit-strength probe 和 failure taxonomy。

---

## 10. 下一轮实验计划

### 10.1 不要直接跑大规模 7-case sweep

当前最优顺序是：

```text
先证明 edit force 能增强；
再做 Pareto sweep。
```

如果直接跑 7-case strength sweep，可能只是再次确认：

```text
背景保持好，但弱编辑 case 目标仍然没出来。
```

### 10.2 Probe 1：弱编辑 / 失败 case

选择：

```text
tshirt_star
tote_leaf
cat_replace_bell_heart_tag
rabbit_sunglasses
```

比较方法：

```text
A. fixed support-v3
B. current rmsgap
C. core-target transport
D. core-target transport + outside lock
E. core-target transport + outside lock + target-prior refresh
F. crop-local version
```

设置：

```text
seed = 10
edit_scale = 1.0
```

目标：

```text
star 是否真正出现；
leaf 是否真正出现；
heart tag 是否可见；
rabbit sunglasses 是否出现且不毁身体。
```

### 10.3 Probe 2：压力成功 case

选择：

```text
red_chair_blue
backpack_remove_toy_charm
backpack_replace_patch_blue
dog_replace_tennis_ball_star
```

比较方法：

```text
current rmsgap
core-target transport
core-target transport + outside lock
target-prior refresh
```

目标：

```text
编辑是否更强；
背景是否仍然稳定；
边界 artifact 是否可控。
```

### 10.4 Probe 成功后再做主 sweep

如果 V2 probe 成功，再跑：

```text
cases:
    7 selected main cases

methods:
    fixed
    rmsgap
    V2 method

edit_scale:
    0.5, 0.75, 1.0, 1.25, 1.5, 2.0

seeds:
    10, 11, 12
```

主分析：

```text
Pareto AUC
best edit under preservation budget
best preservation under edit success threshold
success rate under preservation constraint
```

---

## 11. 论文叙事建议

当前 rmsgap 只能作为 ablation，不适合作为最终主贡献。

新的论文故事应是：

```text
Direct velocity differencing in RF editing under-edits small or weakly localized targets because the target-source flow residual often lacks an object-forming direction. We therefore introduce region-conditioned target transport: the core region follows a stronger target-conditioned RF trajectory, the outside region is locked to the source trajectory, and small targets are edited in a crop-local coordinate frame.
```

中文版本：

```text
直接使用 target-source velocity 差分时，小目标和弱局部目标经常只产生纹理残差，无法形成明确对象。我们提出 region-conditioned target transport：core 区域走更强的 target-conditioned flow，outside 区域锁定 source trajectory，小目标在 crop-local 坐标中编辑。
```

最终 claim 应该从：

```text
small but consistent edit-alignment gain with comparable preservation
```

升级为：

```text
stronger local edit formation under matched preservation constraints
```

或者：

```text
better edit-preserve Pareto frontier through core-target transport and outside trajectory lock
```

---

## 12. 优先级总结

后续方法开发优先级：

```text
1. crop-local editing
2. core target transport
3. outside trajectory lock
4. deterministic target-prior refresh
5. small random refresh
```

原因：

```text
crop-local 直接解决小目标语义太弱；
core target transport 解决 v_target - v_source 太保守；
outside trajectory lock 允许 core 更强编辑而不牺牲背景；
target-prior refresh 解决 core 过度贴 source trajectory；
random refresh 风险最大，最后再做 ablation。
```

一句话总结：

> 下一版不要再做“更聪明地放大 v_target - v_source”，而要让 core 区域在早中期真正进入 target-conditioned flow；小目标则必须 crop-local 放大后编辑，再 blend 回全图。

