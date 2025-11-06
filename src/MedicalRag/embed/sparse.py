"""
中文稀疏表示工具集：

本模块提供两部分能力：
1) 多进程安全、可复用的中文分词流水线（基于 pkuseg），用于高吞吐地将文本切分为 token。
2) 基于 BM25 的稀疏向量化器与轻量级词表管理（token -> id、df、idf 缓存等）。

设计要点：
- 分词器在每个子进程内各自初始化，避免跨进程对象序列化和竞争；通过 Pool(initializer=...) 将模型加载成本摊到进程启动时。
- 词表 `Vocabulary` 既可在建库阶段累积 DF/IDF，也可在查询阶段只读使用；`freeze()` 后会将 idf 预计算为数组以加速访问。
- `BM25Vectorizer` 既支持单进程串行切词，也支持多进程流式切词（`imap`），以平衡吞吐与内存占用。
"""

import math
from typing import List, Dict, Iterable, Iterator
import pkuseg
from multiprocessing import Pool, cpu_count
import os, gzip, pickle, math
from pathlib import Path
from stopwords import stopwords, filter_stopwords

current_dir = Path(__file__).resolve().parent
default_vocab_dir = str(current_dir) + "/vocab/"

# ====== worker 全局 ======
_SEG = None  # 每个子进程里各自持有一个分词器
# 说明：该全局变量仅在子进程内部被初始化与使用。主进程不会共享该实例，
#       从而避免跨进程序列化与锁竞争。通过 Pool 的 initializer 实现“每个
#       子进程自举加载一次”的设计，可显著降低大批量分词时的总体开销。

def _init_seg_worker(domain_model: str):
    """
    每个子进程启动时运行：加载各自的 pkuseg 实例

    参数
    -----
    domain_model: str
        pkuseg 的领域模型名称（例如 "medicine"）。不同的模型在词典、参数上有所差异。

    说明
    -----
    - 将导入放在函数内，可避免主进程与子进程在模块导入阶段的潜在冲突；
    - 通过 `global _SEG` 将分词器实例绑定到当前子进程的模块级全局变量，
      之后该进程内的所有任务都可复用，避免重复加载。
    """
    global _SEG
    import pkuseg as _pk  # 避免主进程/子进程导入冲突
    _SEG = _pk.pkuseg(model_name=domain_model)

def _cut_worker(text: str) -> List[str]:
    """
    子进程真正执行的分词函数

    流程
    ----
    1) 使用已初始化的 `_SEG` 对文本进行切分；
    2) 通过 `filter_stopwords` 过滤停用词；
    3) 去除残留空白并过滤空字符串，返回干净的 token 列表。

    注意
    ----
    必须确保在调用该函数前，当前子进程已通过 `_init_seg_worker` 完成 `_SEG` 初始化。
    """
    toks = filter_stopwords(_SEG.cut(text))
    return [t.strip() for t in toks if t.strip()]


class Vocabulary:
    """维护 token->id 与 id->df，用于稀疏向量化

    职责
    ----
    - 维护词项到整数 ID 的映射（token2id）；
    - 维护每个词项的文档频次 DF（有多少文档出现过该词）；
    - 维护语料规模统计 N（文档总数）与 sum_dl（文档长度总和，用于 avgdl）；
    - 可在 `freeze()` 之后将 idf 预计算并缓存到数组，提升查询阶段的访问速度。

    使用场景
    --------
    - 建库阶段：反复调用 `add_document(tokens)` 累积 DF 与统计量；
    - 查询阶段：只读访问 `idf()` 计算 BM25；如已 `freeze()`，则走数组直取更快。
    """
    def __init__(self):
        self.token2id: Dict[str, int] = {}
        self.df: Dict[int, int] = {}
        self.N: int = 0            # 文档总数
        self.sum_dl: int = 0       # 所有文档长度之和（可选，用于 avgdl）
        # 可选：冻结后缓存
        self.idf_arr: List[float] | None = None

    def add_document(self, tokens: List[str]):
        """向词表加入一篇文档的 tokens，并更新 DF 与统计量。

        说明
        ----
        - `seen` 集合用于确保同一文档内同一词项只计一次 DF；
        - 当词表发生变化（新词加入）时，已缓存的 `idf_arr` 失效，置为 `None`。
        """
        self.N += 1
        self.sum_dl += len(tokens)
        seen = set()
        for t in tokens:
            if t not in self.token2id:
                self.token2id[t] = len(self.token2id)
            tid = self.token2id[t]
            if tid not in seen:
                self.df[tid] = self.df.get(tid, 0) + 1
                seen.add(tid)
        # 词表改变后，旧的 idf 缓存作废
        self.idf_arr = None

    def idf(self, tid: int) -> float:
        """返回指定词项 ID 的 BM25 IDF 值。

        策略
        ----
        - 若已 `freeze()` 并缓存了 `idf_arr`，则直接数组访问（O(1)）；
        - 否则现算：`log(1 + (N - df + 0.5) / (df + 0.5))`。
        """
        if self.idf_arr is not None:
            return self.idf_arr[tid]
        df = self.df.get(tid, 0)
        return math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

    def freeze(self):
        """建库结束后一次性缓存 idf，加速查询

        说明
        ----
        - 预先构造长度为 |V| 的数组，位置即词项 ID，值为对应 IDF；
        - 查询阶段可直接数组访问，避免重复哈希与对数运算。
        """
        V = len(self.token2id)
        df_arr = [0]*V
        for tid, c in self.df.items():
            df_arr[tid] = c
        self.idf_arr = [math.log(1.0 + (self.N - d + 0.5)/(d + 0.5)) for d in df_arr]

    # ---------- 保存 / 加载 ----------
    def save(self, path: str, compress: bool = True):
        """持久化词表到磁盘。

        参数
        -----
        path: str
            目标文件路径或文件名。若未包含 '/', 则会自动保存在当前目录下的 `vocab/` 文件夹中。
        compress: bool
            是否使用 gzip 压缩临时文件再原子替换，默认 True。

        实现细节
        --------
        - 先写入临时文件（.tmp），完成后用 `os.replace` 原子替换，降低中途失败导致文件损坏的风险。
        - `idf_arr` 若存在，也会一并保存，便于快速加载。
        """
        state = {
            "version": 1,
            "token2id": self.token2id,
            "df": self.df,
            "N": self.N,
            "sum_dl": self.sum_dl,
            # 可选缓存，存在就一起存
            "idf_arr": self.idf_arr,
        }
        data = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        
        if '/' not in path:  # 没有自定义绝对路径 , 自动保存在当前目录下的vocab文件夹
            tmp = str(default_vocab_dir) + path + ".tmp"
            path = str(default_vocab_dir) + path
        else:
            tmp = path + ".tmp"
            
        with (gzip.open(tmp, "wb") if compress else open(tmp, "wb")) as f:
            f.write(data)
        os.replace(tmp, path)  # 原子替换，防止中途写坏

    @classmethod
    def load(cls, path_or_name: str):
        """从磁盘加载词表。

        参数
        -----
        path_or_name: str
            绝对/相对路径或简短名称（不含 '/' 时将默认从 `vocab/` 目录拼接加载）。

        返回
        ----
        Vocabulary | None
            加载成功返回 `Vocabulary` 实例；任何异常或路径不存在时返回 None。
        """
        try:
            if '/' not in path_or_name:  # 直接传入的名字
                path = str(default_vocab_dir) + "/" + path_or_name
            if not Path(path).exists:
                return None
            with (gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")) as f:
                state = pickle.load(f)
            v = cls()
            v.token2id = state["token2id"]
            v.df = state["df"]
            v.N = state["N"]
            v.sum_dl = state.get("sum_dl", 0)
            v.idf_arr = state.get("idf_arr", None)
            return v
        except Exception as e:
            return None
        

# ====== 并行分词 + BM25 ======
class BM25Vectorizer:
    def __init__(self, vocab: Vocabulary, domain_model: str = "medicine", k1: float = 1.5, b: float = 0.75):
        # 单进程下仍可直接用
        self.seg = pkuseg.pkuseg(model_name=domain_model)  
        self.domain_model = domain_model
        self.vocab = vocab
        # BM25 参数
        self.k1 = k1
        self.b = b

    # --- 单进程分词 ---
    def tokenize(self, text: str) -> List[str]:
        """在当前进程内用 pkuseg 分词并做基础清洗。

        - 适合轻量/小批量场景；
        - 大批量时建议使用 `tokenize_parallel` 以复用子进程分词器并提升吞吐。
        """
        return [t.strip() for t in filter_stopwords(self.seg.cut(text)) if t.strip()]

    # --- 多进程分词（批处理/流式产出）---
    def tokenize_parallel(
        self,
        texts: Iterable[str],
        workers: int = None,
        chunksize: int = 64
    ) -> Iterator[List[str]]:
        """
        并行分词：按 chunksize 批量发给子进程，流式返回 tokens 列表。

        参数
        -----
        texts: Iterable[str]
            文本可迭代对象，可为生成器，支持大规模流式处理。
        workers: int | None
            进程数，默认 `cpu_count()-1`（至少为 1）。
        chunksize: int
            每个任务块包含的样本数；较大可减少进程间调度开销，较小可降低延迟。

        说明
        ----
        - 使用 `Pool.imap` 实现真正的流式产出，避免一次性加载全部结果到内存；
        - 通过 `initializer=_init_seg_worker` 使每个子进程在启动时各自加载分词模型。
        """
        if workers is None:
            workers = max(1, cpu_count() - 1)
        with Pool(
            processes=workers,
            initializer=_init_seg_worker,
            initargs=(self.domain_model,)
        ) as pool:
            # imap 是流式的，内存占用更稳
            for tokens in pool.imap(_cut_worker, texts, chunksize=chunksize):
                yield tokens

    # --- 从 tokens 构建稀疏向量（避免重复分词） ---
    def build_sparse_vec_from_tokens(
        self,
        tokens: List[str],
        avgdl: float,
        update_vocab: bool = False
    ) -> Dict[int, float]:
        """
        允许传入已分好的 tokens（建议并行切好后再喂这里）

        参数
        -----
        tokens: List[str]
            已分好词的序列。
        avgdl: float
            语料平均文档长度（若不确定，可用构建阶段统计的 `sum_dl / N`）。
        update_vocab: bool
            是否在构建向量同时更新词表（建库阶段可置 True，查询阶段建议 False）。

        返回
        ----
        Dict[int, float]
            稀疏向量，key 为词项 ID，value 为 BM25 权重。

        说明
        ----
        - 构造 TF 映射时容忍 OOV（词不在 `token2id` 中则跳过）；
        - BM25 权重按 `idf * (tf*(k1+1))/(tf + k)` 计算，`k = k1*(1-b + b*dl/avgdl)`。
        """
        if update_vocab:
            self.vocab.add_document(tokens)

        # 查询阶段要容忍 OOV
        tf: Dict[int, int] = {}
        for t in tokens:
            tid = self.vocab.token2id.get(t)
            if tid is None:
                continue
            tf[tid] = tf.get(tid, 0) + 1

        if not tf:
            return {}

        dl = sum(tf.values())
        K = self.k1 * (1 - self.b + self.b * dl / max(avgdl, 1.0))

        vec: Dict[int, float] = {}
        for tid, f in tf.items():
            idf = self.vocab.idf(tid)
            score = idf * (f * (self.k1 + 1.0)) / (f + K)
            if score > 0:
                vec[tid] = float(score)
        return vec if vec is not None else {"0": 0.0}

    # --- 兼容原 API：直接给文本 ---
    def build_sparse_vec(self, text: str, avgdl: float, update_vocab: bool = False):
        """兼容 API：直接输入原始文本，内部先分词再构建稀疏向量。"""
        tokens = self.tokenize(text)
        return self.build_sparse_vec_from_tokens(tokens, avgdl, update_vocab)
    
    def vectorize_texts(self, texts: List[str], avgdl)-> List[Dict[int, float]]:
        """对一组文本逐个构建稀疏向量（串行便利实现）。

        提示：大规模批量时可先用 `tokenize_parallel` 切词，再批量喂给
        `build_sparse_vec_from_tokens` 以减少重复计算和提升吞吐。
        """
        vecs = []
        for i in range(len(texts)):
            tokens = self.tokenize(texts[i])
            vecs.append(self.build_sparse_vec_from_tokens(tokens, avgdl))
        return vecs