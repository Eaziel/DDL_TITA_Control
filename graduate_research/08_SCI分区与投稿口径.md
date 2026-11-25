# SCI、JCR与期刊分区：本项目的投稿口径

核对日期：2026-11-13。用户提出以“SCI二区期刊”为研究目标，分区口径暂未确定；已确认未来没有机器人实机，有Jetson Orin Nano 8GB。本篇解释发表目标，不重新追查培养方案中的最低成果要求。

## 1. 先区分收录、指标和分区

| 常见说法 | 具体含义 | 判断时需要的字段 |
| --- | --- | --- |
| SCI论文、SCI期刊 | 国内通常指被Science Citation Index Expanded（SCIE）收录的期刊及其中收录的文章；SCI原名与SCIE沿革不作为本项目的选刊依据 | 期刊全名、ISSN、收录索引、文章收录状态及时间 |
| Web of Science收录 | 范围更广，其中还包括SSCI、AHCI、ESCI等索引 | 明确记录具体索引名称 |
| JIF或影响因子 | Journal Impact Factor，科睿唯安的期刊引文指标 | 指标年份、具体数值；与CiteScore、SJR、JCI分别记录 |
| JCR Q2 | 指定JCR数据年、指定学科类别中，按相应指标得到的第二四分位 | 通常讨论JIF Quartile；还要核对是否误用了JCI Quartile |
| 中科院2区 | 历史中科院期刊分区表中，指定版本、指定大类或小类的二区 | 发布机构、版本年份、大类/小类及类别名称 |
| Top期刊 | 某套评价表中的Top标记，或一般宣传用语 | 指明哪套表的正式标记；不能直接换算成Q1或一区 |
| EI、CCF A/B/C、学会目录 | 其他收录或推荐体系；期刊和会议也有不同目录 | 每套体系独立核对，不直接换算SCI分区 |

JCR官方界面分别提供索引、数据年、学科、JIF Quartile及JCI Quartile等字段，同一期刊可能有多个学科排名。[JCR使用说明](https://journalcitationreports.zendesk.com/hc/en-gb/articles/28351400868625-Browsing-Journals)

**有影响因子、有Q1/Q2标签，还需另外核实是否属于SCIE。** 科睿唯安已将JIF扩展到ESCI等期刊，并从2024年发布的JCR起将ESCI纳入相应的统一学科排名。因此，用“有IF”替代SCI收录核验会产生错误。[科睿唯安2024年JCR说明](https://clarivate.com/news/clarivate-reveals-worlds-leading-and-trusted-journals-with-the-2024-journal-citation-reports/)

## 2. JCR的Q1—Q4怎样理解

在一个学科类别内，将期刊按JIF从高到低排列，通常可近似理解为：

| 四分位 | 排名位置 |
| --- | --- |
| Q1 | 前25% |
| Q2 | 25%之后至50% |
| Q3 | 50%之后至75% |
| Q4 | 后25% |

并列影响因子会影响排名边界，各组期刊数量未必恰好相同。同一期刊归入两个学科时，可能在一个学科是Q1，在另一个学科是Q2。学校采用最好分区、指定学科分区还是其他口径，需以其实际规则为准。[JCR并列排名解释](https://clarivate.com/academia-government/blog/a-primer-on-ties-in-the-jcr/)

例如，某期刊同时进入“机器人学”和“自动化与控制系统”，两类的竞争期刊与数量不同，所以两个分区可以不同。这个例子用于解释机制，没有指代某本期刊的已核验排名。

影响因子本身可用下式理解：

\[
\mathrm{JIF}_{Y}=\frac{Y\text{年对该刊前两年所发表内容的引用次数}}{\text{该刊前两年发表的可被引项目数}}.
\]

分母主要涉及Article、Review等可被引项目；分子还可能包括对其他文献类型的引用。假设2025年对某刊2023—2024年内容的引用为3000次，相应可被引项目为500篇，则2025 JIF为6.0。这是构造例子。

**发布年份与指标年份要分开。** 例如，2026年发布的年度报告可以给出2025年引文数据；口头所说“2026影响因子”容易遗漏这一层。[JCR指标说明](https://journalcitationreports.zendesk.com/hc/en-gb/articles/28351430970001-Journal-Profile)、[科睿唯安JCR基础指南](https://clarivate.libguides.com/jcr)

JIF描述期刊层面的引文情况，不能据此算出某篇文章的贡献，也不能据此预测本项目的录用概率。[JCR指标使用原则](https://journalcitationreports.zendesk.com/hc/en-gb/articles/28351291240209-JCR-Best-Practices)

## 3. 中科院分区在2026年发生了重要变化

中国科学院文献情报中心于**2026-03-27**公告：从2026年起不再更新与发布期刊分区表，并说明其他机构发布的期刊分区表与该中心无关。[官方声明](https://las.cas.cn/news/tzgg/202603/t20260327_8178738.html)

访问边界：本轮获取了官方域名的检索索引正文；直接打开该网页多次超时。另有高校图书馆发布的转告与原链接相互印证，不能将直接访问写成成功。[安庆师范大学图书馆转告](https://lib.aqnu.edu.cn/info/1542/13254.htm)

这项变化使“最新版中科院二区”尤其需要明确版本。历史分区仍可能用于学校认定，但2029年同济如何采用历史表、后续其他机构的表或本校目录，本轮没有取得可作结论的规则。

历史中科院分区与JCR采用不同的分类和分区方法，不能按四个25%理解其所有版本。大类和小类也分别记录。旧官网帮助页包含过往版本的说明，不能把其中旧学科数量和发布时间当作2026年仍在执行的规则。[历史平台帮助](https://www.fenqubiao.com/user/Help.aspx)

第三方网站显示的“新锐分区”“升级版”“2026分区”，应按页面标明的实际发布者和版本核对。SJR Q2、CiteScore分区也分别属于其他评价体系。它们可以辅助选刊，但不能擅自填入“JCR Q2”或历史“中科院2区”的字段。

## 4. 目前怎样选刊更实际

现阶段把目标写为：**形成面向控制或机器人学习期刊的完整方法论文；以历史中科院二区附近的适配期刊为候选池，正式投稿时核验当年收录及学校采用的评价版本。** 这是一种研究规划口径，尚未指定唯一目标刊。

下面按方法取向筛选，并单列分区证据的局限。表中没有承诺它们在未来均为二区。

| 候选期刊 | 对本项目什么类型的结果较匹配：助手判断 | 分区信息目前核验到哪一步 |
| --- | --- | --- |
| Neurocomputing，ISSN 0925-2312 | 有清楚的学习机制、表示或推断方法，机器人控制用于说明其作用；常规PPO加模块的贡献需充分论证 | 东北师大2025-03-27新闻明确标注2025升级版计算机科学2区、JCR相关学科Q1；属于历史的机构二手证据，未取得该刊当前官方JCR账户记录 |
| ISA Transactions，ISSN 0019-0578 | 状态/扰动估计、约束控制、学习与控制耦合，有充分的分析和闭环评价 | 高校教师资料有历史中科院2区记录，部分新闻混用“中科院JCR2区”；本轮未将这些记录升级为已核验的当前分区 |
| Nonlinear Dynamics，ISSN 0924-090X | 动力学与控制分析构成实质内容，能清楚说明非线性机理、适用条件与方法收益 | 昆明医科大机构库明确区分2025中科院版与2026年发布的JCR，前者工程技术2区、后者相关学科Q1；仍需正式数据库核验 |
| International Journal of Robust and Nonlinear Control，ISSN 1049-8923 | 鲁棒/自适应控制与理论分析；本轮找到其无整机实验的相邻研究先例 | 本轮未核验其符合用户期望的二区口径；在这里用于方法与验证参照 |

范围依据：[Neurocomputing出版社说明](https://shop.elsevier.com/journals/neurocomputing/0925-2312)、[ISA学会期刊说明](https://www.isa.org/standards-and-publications/isa-publications/isa-transactions)、[Nonlinear Dynamics期刊页](https://link.springer.com/journal/11071/aims-and-scope)。Neurocomputing与ISA的ScienceDirect范围页本轮返回403，分别使用出版社商店页与主办学会页。Nonlinear Dynamics网页提取未完整呈现范围段落，方法适配判断还参考其期刊定位，不能声称已完整审阅作者指南。

历史分区线索：[东北师大Neurocomputing新闻](https://ist.nenu.edu.cn/info/1156/5701.htm)、[昆明医科大Nonlinear Dynamics机构库记录](https://ir.kmmu.edu.cn/Document/Detail/FEDC8E2DDA781BF4E966718A7A5495C5.html)、[ISA历史成果资料](https://sice.lnpu.edu.cn/__local/F/58/0B/C44F8DB6E702749A68D4F9022C5_ECC56929_4DEA5.pdf)。这些线索用于缩小选刊范围，不替代学校认定。

与未来预算有关的一项明确变化：Nonlinear Dynamics官方页面公告2027-01-01转为完全开放获取，2026-08-11起收到的稿件若被接收和出版，适用APC及相应豁免规则。本文不记录易变化的收费金额。[出版社公告](https://link.springer.com/journal/11071/aims-and-scope)

## 5. 正式确定目标刊时留下什么记录

一张可核查的记录应包括：期刊名与ISSN、SCIE收录状态及核查日期、JCR数据年与发布年、具体学科和JIF Quartile、历史分区的发布者/版本/大类或小类、学院采用的认定规则、范围与文章类型、开放获取和费用条件。

现在可以推进问题、方法和实验设计。分区口径由导师和学院最终采用的规则决定，不能靠修改论文标题解决；同一项有价值的方法，也可能更适合不同期刊。当前论文复现清单见[03](03_论文复现清单与仓库核查.md)，实验组织及Orin验证边界见[09](09_复现与测试流程.md)。
