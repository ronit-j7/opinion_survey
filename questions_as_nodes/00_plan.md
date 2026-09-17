# Belief Network — How We Build It and What We Test

**Data:** `Survey_Results_UC.csv`. There are 96 rows. 5 are empty, which leaves **n = 91 respondents**. There are **60 statements**, 15 each on Technology (T), Education (E), Society (S) and Environment (V).

**Status:** plan only. Nothing below has been run yet.

---

## Part 1 — How the graph is built

### 1a. Nodes

**Each node is one survey statement.** That gives 60 nodes, and the nodes are not people.

Answers are coded SD = 1, D = 2, N = 3, A = 4, SA = 5. "No Comments" and blank answers count as missing.

Each node keeps its theme label T/E/S/V. The label is **never used to build the graph**. It is used only afterwards, to check whether the graph rediscovers the themes (Test 6).

> **Why not merge statements into hand-made groups first?**
> A group score, such as the mean of the "AI in education" statements, has more distinct values and more variance than a single statement. It also cuts the number of possible edges from 1,770 to about 100.
> That only helps if the statements in a group actually move together. We checked 15 intuitive groups using Cronbach's α. Only 1 reached the usual 0.70 bar, and several were below 0.45. For example, "AI optimism" (T01, T05, T06, T15) had α = 0.32.
> Averaging statements that don't co-vary produces a blurry node, not a more precise one. So we keep single statements as nodes, and we **test** the hand-made groups instead (Test 7).

### 1b. Edges

**An edge between statements $j$ and $k$ is their partial correlation $\omega_{jk}$.** It measures how strongly two statements are linked after the other 58 are held fixed.

- **Undirected, weighted, signed.** The weight is $\omega_{jk} \in [-1, 1]$. A positive edge means people who agree with one statement tend to agree with the other. A negative edge means agreeing with one goes with disagreeing with the other.
- **No edge** means $\omega_{jk} = 0$. The two statements are independent once everything else is accounted for. Any link between them runs through other statements.

Why partial and not plain correlation? Suppose A–B and B–C are both strong. Then A and C will also correlate, purely through B. A plain correlation network draws an A–C edge anyway. The partial correlation network doesn't.

### 1c. How edges are computed

**Step 1 — Correlation between every pair of statements.**

Take the respondents who answered both $j$ and $k$. Rank them by their answer to $j$, and separately by their answer to $k$. Tied people share the average of the positions they occupy. Spearman's $r^S_{jk}$ is the Pearson correlation of these two rank lists.

Then convert it to the normal-scale correlation that the model in Step 3 expects:

$$\hat r_{jk} = 2\sin\!\left(\frac{\pi}{6}\, r^S_{jk}\right)$$

This changes values by at most 0.018. Collect all pairs into a 60×60 matrix $R$.

**Step 2 — Make $R$ valid.** Each entry uses a slightly different set of people, so $R$ may not be positive definite. If it isn't, replace it with the nearest valid correlation matrix: clip negative eigenvalues to a small $\varepsilon$, then rescale the diagonal back to 1. (Our current $R$ has smallest eigenvalue 0.0013, which is valid but close to the edge.)

**Step 3 — Sparse precision matrix (graphical lasso).** In a normal model, partial correlations come from the inverse correlation matrix $\Theta = R^{-1}$. With 91 people and 1,770 pairs, a plain inverse is mostly noise. So we estimate $\Theta$ with a penalty that sets weak entries to exactly zero:

$$\hat\Theta_\lambda = \arg\max_{\Theta \succ 0} \Big[\log\det\Theta - \operatorname{tr}(R\,\Theta) - \lambda \sum_{j \neq k} |\theta_{jk}|\Big]$$

A larger $\lambda$ gives fewer edges.

**Step 4 — Choose $\lambda$.** Try 100 values of $\lambda$. Pick the one that minimizes the extended BIC:

$$\mathrm{EBIC}(\lambda) = -n\Big[\log\det\hat\Theta_\lambda - \operatorname{tr}(R\,\hat\Theta_\lambda)\Big] + |E_\lambda|\log n + 4\gamma\,|E_\lambda|\log p$$

- $|E_\lambda|$ is the number of edges.
- $n = 85$, the median number of people per pair.
- $p = 60$.
- $\gamma = 0.5$.

The first term rewards fit. The other two charge a price for every extra edge.

**Step 5 — Convert to edge weights.**

$$\omega_{jk} = -\frac{\hat\theta_{jk}}{\sqrt{\hat\theta_{jj}\,\hat\theta_{kk}}}$$

Every non-zero $\omega_{jk}$ becomes an edge.

**Baseline graph (for Test 2).** This graph uses a plain correlation edge wherever $\hat r_{jk}$ is significant. Significance comes from $t = \hat r\sqrt{(n_{jk}-2)/(1-\hat r^2)}$, with Benjamini–Hochberg correction at 5% across all 1,770 pairs.

---

## Part 2 — What we learn from the graph

Each test states the question, the method, and what each possible outcome would mean.

### Test 1 — Where does the class agree, disagree, or split?

This test uses the answers directly, not the graph. It explains why some nodes end up isolated.

**How:** for each statement, compute:

- the agree share $a_j = P(x \geq 4)$,
- the disagree share $d_j = P(x \leq 2)$,
- the standard deviation $\sigma_j$.

**Outcomes:**
- **Agreement consensus** ($a_j \geq 0.85$, low $\sigma_j$). 19 statements, e.g. E15 and V13.
- **Disagreement consensus** ($d_j \geq 0.85$). None. The closest is E03 (compulsory attendance) at 70%.
- **Split** (both $a_j$ and $d_j$ large, high $\sigma_j$). E02 (exams measure knowledge) is the clearest case: 41% disagree, 32% agree.

Consensus statements have little variance. They will have few or no edges. That means *everyone agrees*, not *this belief is unimportant*.

### Test 2 — Which links are direct?

**How:** compare edge sets. The explained-away share is $1 - |E_{\text{partial}} \cap E_{\text{plain}}| \,/\, |E_{\text{plain}}|$.

**Outcomes:**
- **High share (e.g. > 50%).** Most correlations in the class run through a few key beliefs. The belief system is organised around a small core.
- **Low share.** Beliefs are linked pairwise and directly, with no central organiser.

### Test 3 — Which beliefs are hubs?

**How:**
- **Strength:** $s_j = \sum_k |\omega_{jk}|$.
- **Expected influence:** $EI_j = \sum_k \omega_{jk}$, which keeps signs.

Both are reported as z-scores. Closeness and betweenness are reported only if they pass Test 9.

**Outcomes:**
- **One or a few clear hubs.** For example, T01 "AI improves society" might tie the tech beliefs together. Changing that one opinion would be associated with changes in many others.
- **Flat strength distribution.** No belief anchors the rest.
- **Strength high but EI near 0.** The hub has both positive and negative links. It pulls some beliefs one way and others the opposite way. E02 or E03 are candidates.

### Test 4 — Which beliefs bridge themes?

**How:** bridge strength $b_j = \sum_{k:\,\text{theme}(k) \neq \text{theme}(j)} |\omega_{jk}|$.

**Outcomes:**
- **High $b_j$.** The statement connects domains. For example, T12 (AI regulation) linking to Society statements would show students think about AI as an ethics question.
- **All $b_j$ low.** Students keep the four domains separate.

### Test 5 — How well is each belief explained by its neighbours?

**How:** predictability $R^2_j$, the variance of statement $j$ explained by regressing it on its graph neighbours.

**Outcomes:**
- **High $R^2_j$.** The belief follows from the others. It is embedded in the system.
- **Low $R^2_j$.** The belief stands alone. Either it is a consensus item (Test 1), or it is driven by something outside this survey.

### Test 6 — Does the graph rediscover the four themes?

**How:**
1. Find communities using walktrap, as in Exploratory Graph Analysis. Louvain serves as a check. Both run on the positive edges only.
2. Compare the communities $U$ with the themes $\mathcal{T}$ using NMI and ARI:

$$\mathrm{NMI} = \frac{2\,I(U;\mathcal T)}{H(U)+H(\mathcal T)}$$

3. Separately, test whether the themes form an unusually modular split. Compute the modularity $Q$ of the theme partition. Compare it against 10,000 random relabellings into four groups of 15, which gives a p-value.

**Outcomes:**
- **NMI high, $p < 0.05$.** Students' opinions are organised the way the survey was designed. Topic is what structures belief.
- **NMI low, but communities still clearly modular.** Opinions are organised along some other axis that cuts across topics. For example, a "trust in institutions" cluster might mix T, S and V items. This is arguably the more interesting finding.
- **No clear communities** (low $Q$ for every partition). Beliefs don't form clusters. The graph is a diffuse web.

### Test 7 — Do our hand-made groups hold up?

This test uses the grouping idea from 1a.

**How:** define groups *before* looking at the graph, such as "AI in education" or "climate policy". Then compute:
- modularity $Q_{\text{groups}}$, with the same permutation test as Test 6,
- NMI with the found communities.

**Outcomes:**
- **$Q_{\text{groups}} > Q_{\text{themes}}$.** Finer topics describe belief structure better than the four broad themes.
- **Groups not modular.** Statements that *look* related to us aren't linked in students' answers. This agrees with the low Cronbach's α values we already found.

### Test 8 — Which themes are linked to each other?

**How:** a 4×4 matrix of mean $|\omega|$ within and between themes:

$$\bar W_{ab} = \operatorname{mean}_{j \in a,\, k \in b} |\omega_{jk}|$$

**Outcomes:**
- **Strong diagonal.** Each theme is internally coherent.
- **Strong off-diagonal cell,** e.g. T–S. Opinions on those two domains move together.
- **Uniform matrix.** Theme boundaries don't matter.

### Test 9 — Can we trust any of this?

**How:**
- **Edge bootstrap.** Resample the 91 people 1,000 times and rebuild the graph each time. Record each edge's 95% interval and how often it appears.
- **Case-dropping.** Drop 5%–75% of people and recompute centrality. Find the largest drop fraction at which the ranking still correlates ≥ 0.7 with the full-data ranking in 95% of runs. That fraction is the **CS-coefficient**.
- **Sensitivity.** Rerun Tests 3 and 6 with each change below:
  - polychoric correlation instead of Spearman,
  - complete cases only ($n = 68$),
  - $\gamma \in \{0, 0.25\}$.

**Outcomes:**
- **CS ≥ 0.5.** The centrality ranking is solid, and we can name hubs confidently.
- **0.25 ≤ CS < 0.5.** Report hubs with a caution.
- **CS < 0.25.** Don't interpret the ranking. Report only edges that appear in most bootstraps, plus the community results, if those are stable.

With n = 91, the last case is a real possibility. Reporting it honestly is still a valid result.

### Optional — Test 10: Is there a coherent minority view?

**How:** build a Response Item Network (ResIN). Each node is a statement–answer pair, e.g. "T01 = Disagree". The edge weight is the φ correlation between two such pairs, keeping positive edges only.

**Outcomes:**
- **Disagree-answers cluster together.** A consistent sceptic minority exists. For example, the same few people doubt AI across several statements.
- **Disagree-answers scattered.** Dissent is idiosyncratic. It comes from different people on different statements.
