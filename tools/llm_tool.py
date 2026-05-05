"""LLM helpers: Gemini embedding + chat completions wrappers."""
import os
import time
from typing import List, Optional

import numpy as np
from openai import OpenAI


def _get_client(api_key: Optional[str] = None):
    from google import genai
    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "GOOGLE_API_KEY not set. Export it before calling embed_batch:\n"
            "  export GOOGLE_API_KEY=<your-google-genai-key>"
        )
    return genai.Client(api_key=key)


def _nonempty_text(t) -> str:
    """保证返回非空字符串，否则 API 会报 The text content is empty."""
    if t is None:
        return "not available"
    s = str(t).strip()
    if not s or s.lower() == "nan":
        return "not available"
    return s


def embed_batch(
    texts: List[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    output_dimensionality: int = 3072,
    api_key: Optional[str] = None,
) -> List[List[float]]:
    """
    调用 Gemini 对一批文本做 embedding。空/None/nan 会替换为 "not available"，避免 API 报 INVALID_ARGUMENT。
    """
    from google.genai import types
    texts = [_nonempty_text(t) for t in texts]
    client = _get_client(api_key)
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=output_dimensionality,
        ),
    )
    return [e.values for e in result.embeddings]


def embed_all(
    texts: List[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    output_dimensionality: int = 3072,
    batch_size: int = 50,
    sleep_between_batches: float = 0.5,
    api_key: Optional[str] = None,
    progress_bar: bool = True,
) -> np.ndarray:
    """
    对全部 texts 按 batch 调用 Gemini embedding，返回 (N, dim) 的 numpy 数组。
    progress_bar：是否用 tqdm 显示进度。
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    all_emb = []
    n_total = len(texts)
    it = range(0, n_total, batch_size)
    if progress_bar and tqdm is not None:
        it = tqdm(it, total=(n_total + batch_size - 1) // batch_size, unit="batch", desc="embed")
    for i in it:
        batch = texts[i : i + batch_size]
        embs = embed_batch(
            batch,
            task_type=task_type,
            output_dimensionality=output_dimensionality,
            api_key=api_key,
        )
        all_emb.extend(embs)
        if sleep_between_batches > 0:
            time.sleep(sleep_between_batches)
    return np.array(all_emb, dtype=np.float32)


def embed_all_with_checkpoint(
    texts: List[str],
    checkpoint_path: str,
    task_type: str = "RETRIEVAL_DOCUMENT",
    output_dimensionality: int = 3072,
    batch_size: int = 50,
    sleep_between_batches: float = 0.5,
    api_key: Optional[str] = None,
    progress_bar: bool = True,
) -> np.ndarray:
    """
    带断点恢复的 embedding：用 memmap 增量写入，每批只写当前 batch，不会越写越慢。
    下次运行从未完成处继续。兼容旧版 .npz checkpoint（会迁移到增量格式）。
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    n_total = len(texts)
    dim = output_dimensionality
    base = os.path.splitext(checkpoint_path)[0]
    emb_path = base + "_emb.dat"
    n_done_path = base + "_n_done.npy"

    start_idx = 0
    if os.path.isfile(emb_path) and os.path.isfile(n_done_path):
        start_idx = int(np.load(n_done_path).item())
        if start_idx >= n_total:
            memmap = np.memmap(emb_path, dtype=np.float32, mode="r", shape=(n_total, dim))
            return np.array(memmap[:n_total], copy=True)
        if progress_bar and tqdm is not None:
            tqdm.write(f"Resume from {start_idx}/{n_total}")
        memmap = np.memmap(emb_path, dtype=np.float32, mode="r+", shape=(n_total, dim))
    elif os.path.isfile(checkpoint_path):
        with np.load(checkpoint_path, allow_pickle=True) as data:
            if "emb" in data:
                emb_arr = data["emb"]
                start_idx = emb_arr.shape[0]
                if start_idx >= n_total:
                    return np.array(emb_arr, dtype=np.float32)
                memmap = np.memmap(emb_path, dtype=np.float32, mode="w+", shape=(n_total, dim))
                memmap[:start_idx] = emb_arr[:start_idx]
                np.save(n_done_path, np.array(start_idx))
                if progress_bar and tqdm is not None:
                    tqdm.write(f"Migrated .npz -> incremental, resume from {start_idx}/{n_total}")
            else:
                memmap = np.memmap(emb_path, dtype=np.float32, mode="w+", shape=(n_total, dim))
    else:
        memmap = np.memmap(emb_path, dtype=np.float32, mode="w+", shape=(n_total, dim))

    num_remaining_batches = (n_total - start_idx + batch_size - 1) // batch_size
    it = range(start_idx, n_total, batch_size)
    if progress_bar and tqdm is not None:
        it = tqdm(it, total=num_remaining_batches, unit="batch", desc="embed")
    for i in it:
        end_i = min(i + batch_size, n_total)
        batch = texts[i:end_i]
        embs = embed_batch(
            batch,
            task_type=task_type,
            output_dimensionality=dim,
            api_key=api_key,
        )
        memmap[i:end_i] = np.array(embs, dtype=np.float32)
        np.save(n_done_path, np.array(end_i))
        if sleep_between_batches > 0:
            time.sleep(sleep_between_batches)
    return np.array(memmap[:n_total], copy=True)


def openai_textgen(prompt: str, model: str = "gpt-4o", temperature: float = 0.7, max_tokens: int = 100) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set; export it before calling openai_textgen.")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip().strip('\'"""')
