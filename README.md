# Minecraft Art Studio

一个从自然语言生成 Minecraft 模组美术资源的通用流水线。它会从本地原版资源索引中选择参考，规划资源类别与视觉设计，生成像素贴图，进行结构验证、盲审和目标评审，并输出资源包。

## 使用

在 PowerShell 中配置模型后，直接传入目标即可：

```powershell
$env:LLM_API_KEY = (Get-Content 'C:\Users\GMH13\Desktop\dsapikey.txt' -Raw).Trim()
$env:LLM_BASE_URL = 'https://api.deepseek.com/v1'
$env:LLM_MODEL = 'deepseek-flash'
$env:LLM_REASONING_EFFORT = 'high'

.\.venv\Scripts\python.exe -m studio_next generate `
  --query "血凝胶" `
  --out outputs\blood_gel `
  --rounds 3 `
  --max-repairs 1
```

输出目录包含最终贴图、预览和资源包。`--form auto` 为默认值；流水线会在物品、方块、实体和交叉植物贴图之间自动选择。

## 重建原版资源索引

首次使用其他 Minecraft 安装目录时执行一次：

```powershell
.\.venv\Scripts\python.exe -m studio_next index-vanilla `
  --source 'C:\path\to\.minecraft\versions\1.12.2-Forge_14.23.5.28641'
```

索引写入 `references/index.json`，图片缓存写入 `references/.cache/`。日常生成会直接使用它们。

注意：索引把素材源的绝对路径写进了 `source` 与每条 `source_locator`，而指纹只反映内容。
换了路径（例如从 Windows 搬到 WSL）后必须加 `--rebuild`，否则会命中缓存并把旧路径原样写回，
生成时才在 `materialize` 阶段报找不到 JAR。

## 查看逻辑资源名（一个名字 = 该方块/实体的全部贴图）

一个方块在资源目录里是若干张贴图（`log_oak` / `log_oak_top`）。
`list-groups` 按 blockstates → models → textures 把它们合并成一个可选的名字，
并且**不读取任何图片**（只解析 JSON），原版 1.12.2 约 0.8 秒、AoA3 约 1.8 秒：

```bash
python3 -m studio_next list-groups \
  --source "/mnt/c/.../versions/1.12.2-Forge_14.23.5.28641" \
  --filter oak_log
#   minecraft:block/oak_log   [block] 2 tex: log_oak, log_oak_top
#   minecraft:item/oak_log    [item]  2 tex: log_oak, log_oak_top

# 叠加多个来源：后面的覆盖前面的（等同于资源包叠加顺序）
python3 -m studio_next list-groups \
  --source "/mnt/c/.../versions/1.12.2-Forge_14.23.5.28641" \
  --source "/path/to/your-mod/src/main/resources" \
  --category block --namespace yourmod --limit 20

# 选定一个名字之后，才把这些贴图取出来
python3 -m studio_next list-groups \
  --source "/mnt/c/.../versions/1.12.2-Forge_14.23.5.28641" \
  --extract minecraft:block/crafting_table --to outputs/_refs

# 导出 router 用的清单
python3 -m studio_next list-groups --source "<jar>" --json > catalogue.json
```

把「原版 + 正在开发的模组资源目录」叠加起来取参考，新生成的美术就会对齐模组既有风格，
而不是每开一次就换一套配色。

已知的两处启发式（都有测试覆盖）：

* **实体**：1.12.2 的实体→贴图映射编译在渲染器 Java 代码里，jar 中没有数据，
  因此按目录/名字前缀聚合（`horse_black.png` → `horse`）。
* **代码驱动的帧族**：`clock_00..63`、`bow_pulling_0..2` 是真实存在的 model 文件，
  会被折叠回 `clock` / `bow` 一个名字；而 `record_11` / `record_13` 是真正的两个物品，
  不会被错误合并。

## 动态参考（不依赖持久化索引）

`generate` 可以直接从资源根现场解析参考，不再依赖 `references/index.json`：

```bash
python3 -m studio_next generate \
  --query "血凝胶" \
  --source "/mnt/c/.../versions/1.12.2-Forge_14.23.5.28641" \
  --source "/path/to/your-mod/src/main/resources" \
  --rounds 3
```

- `--source` 可重复，后面的覆盖前面的（资源包叠加顺序）。把原版和你正在开发的模组叠在一起，
  新生成的美术就会对齐模组既有风格，而不是每开一次就换一套配色。
- 路由只看到**逻辑名字**（`oak_log`、`bow`、`iron_helmet`），看不到 `log_oak_top`。
- 选中一个名字后，它名下的**全部**贴图一起作为参考送进规划阶段
  （`bow` → `bow_standby` + `bow_pulling_0/1/2`，共 4 张）。
- 文本特征按**内容哈希**缓存（`references/.cache/live/text/`）：原图一改自动重算，
  文件改名、换机器、换挂载点仍然命中。
- `--recall-limit N` 可以把送进 router 的名字数量截断（默认送完整目录）。

### 提示词里的名字分类

router 看到的每行是 `名字 类别/同类词根`：

```
iron_helmet item/helmet
chainmail_helmet item/helmet
diamond_helmet item/helmet
iron_sword item/sword
```

「同类词根」是**从当前目录动态统计出来的**——某个名字的最后一个词被至少两个资源共享时才算——
不是硬编码表；列表按它排序，相近的资源会连续出现，便于模型对比选择。
斜杠后面只是分组标签，永远不是名字的一部分。

## 生成一族图片

一次请求生成一组必须共享同一视觉语言的贴图（盔甲套、工具套）：

```bash
# 成员由模型从请求里推导
python3 -m studio_next generate --query "铁质盔甲套装" --family --max-members 4 --source "<jar>"

# 或者显式指定成员
python3 -m studio_next generate --query "铁质盔甲" --family \
  --family-members "helmet,chestplate,leggings,boots" --source "<jar>"
```

工作方式：

1. `plan_family` 把请求拆成一个有序成员表，每个成员是一个**完整**的自然语言请求。
2. 第一个成员走完整流水线，产出**风格锚点**。
3. 其余成员走同一条流水线，但额外挂上锚点图（角色 `pixel_style,palette`），
   保证配色、材质读感、描边与高光处理一致。
4. 锚点的**调色板斜坡**在第一遍就套到每个后续成员身上：模型保留自己的色板命名与
   明暗顺序，颜色取值归全族所有。只对"命名色板"生效是不够的——一次实跑里四帧中
   有三帧把十六进制字面量直接写进了部件，锚点斜坡一个像素都没碰到，同一把弓出了四套配色。
   现在命名色板和字面量都会按亮度排名落到锚点斜坡上。
5. 每个成员有独立的输出目录与质量报告；`family.json` 记录拆分计划、逐成员结果，
   以及**确定性的调色板重合度**（不靠模型自评）：低于 `minimum_overlap`（默认 0.5）
   的成员会被标成 `consistent: false`，风格漂移一眼可见。

### 同一对象的多个状态（弓的四帧、时钟的表盘）

`family.json` 里还有两样东西专门针对"同一个物体、不同状态"：

- `inherited_shape_modes`：轮廓关系是**物体级**属性，不是单帧的。一次水晶弓实跑里，
  同一把弓的四帧分别声明了 `appearance_only`、`local_silhouette_edit`、
  `local_silhouette_edit`、`preserve_silhouette`，两帧因此保留了手绘的团块轮廓。
  现在第一帧对某个参考组（`group=`）的判定会传给同组的后续帧；盔甲套这类成员来自
  不同参考组的集合不受影响。
- `inherited_anchor_paint`：**画法继承**。光统一颜色取值不够——一次水晶弓实跑里四帧各自解出
  不同的明暗分层（锚点弓身 6 档、第二帧只有 4 档且全在暗端），同一把弓读起来像四种材质。
  现在同一组参考、且**部件掩码落在同样像素上**的成员，直接继承锚点对应部件的整份画法：
  色阶、明暗轴、噪点、高光比例、标记，以及锚点手绘的 pixel_map。匹配只看像素重叠，
  不看部件名（bow_body / crystal_bow_body 指的是同一块区域）；重叠也是护栏——
  盔甲套这种落在不同图集上的成员什么都匹配不上，各画各的。
  实测帧间精确一致度 0.043 → 0.144，容差内一致度 0.126 → 0.252。

### 帧数很多的名字

一个逻辑名可以拥有几十帧：原版 clock 有 64 帧、compass 有 32 帧。全部附上会把整份视觉预算
花在同一个物体的近似副本上。现在一个名字最多展开 `_MAX_GROUP_FRAMES`（默认 8）帧：
**本次请求对应的那一帧**，加上其余状态的均匀取样。帧序号仍是源家族里的真实序号，
所以 `member=37/64` 这类信息不会因为切片而失真。
- `continuity`：**帧间一致性**的确定性度量。调色板重合度回答不了这个问题——同一把弓
  的两帧可以共享色板而画的是两把不同的弓。这里比的是两帧共同身体上的逐像素一致度，
  并区分两个会以不同方式失败的数：
  - `agreement`（精确一致）：同一个像素有没有被画成同一种颜色。
  - `near_agreement`（容差内一致，默认每通道 24）：至少是不是同一套颜色语言。

  判定是**相对的**：原版弓自己的四帧两两也只有 29%–66% 的精确一致度（弦会动、弓臂会微弯，
  细长形状让每个移动的像素都显眼）。所以基线用这次实跑真正参考的那几帧原图，
  而不是拍一个绝对阈值——绝对阈值会让原版美术自己不及格。

成员之间失败互相隔离：某个成员报错不会丢掉其余成员的结果。


## 轮廓锁定（默认模式）

当请求只是**换材质/换配色**时，参考图的 alpha 就是答案，设计应该发生在"怎么画"上，
而不是"画成什么形状"。这条路径以前不可靠：模型会亲手把参考的轮廓重画一遍 ASCII 掩码，
画歪是常态。

一次水晶弓实跑里，描述符写的是 `appearance_only`，`reference_strategy` 说
"用 bow_standby 的 alpha 当轮廓宿主"，然后模型交回来一坨手绘方块——后面每个阶段都
忠实地给它上了色。

现在：

1. `shape_edit_mode` 是结构化的字段，优先于对 `reference_strategy` 的关键词扫描。
   `appearance_only` / `preserve_silhouette` 直接锁定；`new_silhouette` 直接不锁；
   `local_silhouette_edit` 只有在策略里**点名了要改哪里**（断裂、缺口、重设计……）
   时才不锁——"在某个命名宿主上做最小 alpha 编辑"如果没命名宿主，就是没事要改。
2. 不回退到要求模型手画轮廓。`reference_silhouette_partition` 直接拿**模型自己画的部件掩码**
   当归属证据：把源 alpha 的每个像素判给掩码离它最近的那个部件。轮廓和源图逐像素相同，
   而模型对"哪个部件在哪"的划分被完整保留，后面的 AppearanceSpec 仍然对得上。
   用最近而不是相交，是因为手绘草稿常常差一格：同一把水晶弓的草稿把弦画在第 13 列，
   源图的弦在第 12 列，相交判定会一个弦像素都找不到而拒绝执行一个完全清楚的意图。
3. 同尺寸参考的挑选是**认名字**的。一次弓的请求会带上全部四帧，`bow_pulling_0` 排在第一个；
   按列表顺序选会让待机帧去套半拉开帧的轮廓，家族要的动画就没了。现在按请求名匹配
   （`crystal_bow_standby` 也能匹配到源帧 `bow_standby`），匹配不上才退回列表顺序。

带 UV 图集的形态（entity_uv / block_multi）不走这条路：它们已经有经过验证的 alpha 权威，
这里不去越俎代庖。

`--free-silhouette` 仍然存在：那才是"让模型自己的方案决定轮廓"的开关。

## 自查脚本

```bash
python3 tools/review_family.py outputs/<run>/family.json   # 逐帧对源图的轮廓 + 帧间一致性
python3 tools/magnify.py '{"out":"/tmp/x.png","entries":[["标签","路径"]]}'   # 放大对比图
python3 tools/continuity.py a.png b.png c.png              # 单独看两两帧一致度
```
