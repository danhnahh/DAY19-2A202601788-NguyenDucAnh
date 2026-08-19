# Phân Tích Ca Lỗi (Root-Cause Analysis) — Lab 19

**Học viên:** Nguyễn Đức Anh · **Mã:** 2A202601788
**Nguồn dữ liệu:** `outputs/graphrag_eval_results.csv` (25 câu), `outputs/golden_coverage.csv`, `outputs/entity_resolution_audit.csv`

> **Bối cảnh số liệu:** 25 câu hỏi được chấm, **12 câu cho điểm hợp lệ**, 13 câu trả về exception. Bản thân 13 exception đó là Ca lỗi 1 và 3 dưới đây, nên chúng được phân tích chứ không bị giấu đi.

---

## Ca 1 — Lỗi im lặng ở tầng pipeline: `pandas.Series.name` che mất cột `name`

Đây là ca lỗi nghiêm trọng nhất vì nó **không crash ngay**, mà trả về dữ liệu sai rồi mới crash ở một chỗ khác cách đó ba cell.

| | |
|---|---|
| **Triệu chứng đầu tiên** | `TypeError: sequence item 1: expected str instance, int found` tại cell 4.1c, khi `", ".join(s["name"] for s in seeds)` |
| **Triệu chứng thứ hai** | Sau khi sửa vội: `NameError: name 'r' is not defined` ở 10/25 câu trong cell 4.3 |
| **Số câu bị ảnh hưởng** | 10/25 (G5000-26, 27, 34, 35, 38, 39, 40, 42, 48, 49) |

**Đoạn code gốc** (cell 3.2, `match_seeds()`):

```python
r = entity_match_store.iloc[int(idxs[j])]
matched.append({"id": r.id, "name": r.name, "type": r.type})
```

**Truy vết nguyên nhân gốc rễ:**

1. *Triệu chứng:* seed khớp bằng vector trả về `name` là số nguyên thay vì tên thực thể.
2. *Kiểm chứng trực tiếp:*
   ```python
   r = pd.DataFrame([{"id":"x1","name":"Amazon","type":"Company"}]).iloc[0]
   r.name      # -> 0          (nhãn index của dòng)
   r["name"]   # -> 'Amazon'   (cột name)
   ```
3. *Nguyên nhân gốc:* `entity_match_store.iloc[k]` trả về một **pandas Series**, mà `Series.name` là **thuộc tính dựng sẵn** chứa nhãn index — nó che hoàn toàn cột `"name"`. Trong khi đó `r.id` và `r.type` không trùng thuộc tính nào của Series nên vẫn lấy đúng cột. Kết quả là lỗi chỉ ảnh hưởng **đúng một trường**, và chỉ trên **đúng một nhánh** (fuzzy match), nên không có test nào bắt được.
4. *Vì sao hệ thống không sập ngay:* graph traversal dùng `seed["id"]` (đúng) chứ không dùng `seed["name"]`, nên BFS vẫn chạy bình thường suốt Phần 3. Dữ liệu sai chỉ lộ ra khi bảng chẩn đoán ở 4.1c cố ghép chuỗi.
5. *Nguyên nhân của triệu chứng thứ hai:* khi sửa tay trên Colab, **dòng gán `r` bị xoá mất** cùng lúc với việc thay `r.name` bằng `r["name"]`. Code còn lại trong notebook đã nộp:

   ```python
   if float(sims[j]) >= fuzzy_threshold:
       matched.append({"id": r["id"], "name": r["name"], "type": r["type"]})
       # thiếu hẳn dòng: r = entity_match_store.iloc[int(idxs[j])]
   ```

   Hệ quả: câu hỏi nào có seed **khớp fuzzy thành công** (`sims[j] >= 0.66`) sẽ chạm `r` khi `r` chưa bao giờ được gán → `NameError`.

**Bằng chứng cho kết luận số 5:** toàn bộ 10 câu `NameError` đều có `matched_seeds = 0` trong `golden_coverage.csv`, và output của cell 4.1c in ra đúng **10 dòng** `seed check loi: NameError name 'r' is not defined` — cell này bọc `match_seeds()` trong `try/except` nên nó nuốt exception và ghi 0. Ngược lại, những câu `matched_seeds = 0` mà **không** lỗi ở 4.3 (G5000-29, 33, 37, 41, 43, 46, 47) là các câu `extract_seeds()` không trả về seed nào, nên vòng lặp không bao giờ chạy vào nhánh fuzzy.

**Hệ quả bị đánh giá thấp — và đây là điểm quan trọng nhất của ca lỗi này:** vì `NameError` chỉ nổ khi `sims[j] >= 0.66`, **10 câu bị lỗi chính là 10 câu mà fuzzy matching ĐÃ thành công**. Nói cách khác, đây đúng là những câu GraphRAG có seed thật và có cơ hội thắng. Bảng benchmark hiện tại (`NO_SEED` 7/12, seed coverage 8/25) vì vậy **đánh giá thấp GraphRAG một cách có hệ thống**: nó chỉ chứa những câu mà graph khớp *chính xác* qua Cypher hoặc không khớp gì cả, còn toàn bộ nhánh khớp gần đã bị loại khỏi mẫu. Sau khi khôi phục một dòng code, độ phủ seed dự kiến tăng từ 8/25 lên khoảng 18/25.

**Khắc phục:**

```python
r = entity_match_store.iloc[int(idxs[j])]
matched.append({"id": r["id"], "name": r["name"], "type": r["type"]})
```

**Bài học phòng vệ:** không dùng attribute access trên pandas Series cho các tên trùng API của pandas (`name`, `index`, `values`, `size`, `shape`, `dtype`). Với dữ liệu do người dùng đặt tên cột, luôn dùng `r["name"]`.

---

## Ca 2 — GraphRAG thua Flat RAG vì hybrid context **giảm** vector recall

| | |
|---|---|
| **Question ID** | G5000-50 |
| **Nhóm** | multi-hop |
| **Câu hỏi** | *Compare the chip-related AI positioning of NVIDIA, AMD, and Intel in the selected 5,000-row scope. What distinct signal is reported for each?* |
| **Điểm Flat RAG** | Comp **5** / Faith **5** / Multi-hop **5** |
| **Điểm GraphRAG** | Comp **2** / Faith **4** / Multi-hop **3** |
| **Chẩn đoán** | `matched_seeds = 1` (Advanced Micro Devices), `collected_edges = 8`, `no_seed = 0` |

**Flat RAG trả lời:** đủ cả ba tín hiệu — NVIDIA (nhu cầu chip cho chatbot tăng vọt, dự báo doanh số vượt kỳ vọng), AMD (AWS đang cân nhắc chip AI mới), Intel (3D V-Cache).

**GraphRAG trả lời:** *"1. **NVIDIA**: The context does not provide specific information about NVIDIA's AI positioning."* — mất hoàn toàn một trong ba thực thể được hỏi.

**Truy vết nguyên nhân gốc rễ:**

| Mắt xích | Kiểm tra | Kết luận |
|----------|----------|----------|
| Seed extraction | `matched_seeds = 1` | Chỉ khớp AMD; NVIDIA và Intel **không** có seed |
| Seed resolution | fuzzy `>= 0.66` | Không khớp node NVIDIA nào |
| Extraction sót cạnh | đồ thị 209 node / 124 cạnh | Chunk chứa tin NVIDIA nằm ngoài 400 chunk gửi qua LLM |
| Super-node cắt mất | `supernode_events = 0` | Không phải nguyên nhân |
| Chạm `GLOBAL_EDGE_CAP` | `collected_edges = 8` ≪ 250 | Không phải nguyên nhân |
| Tràn `MAX_GRAPH_CONTEXT_CHARS` | 8 cạnh ≈ vài trăm ký tự | Không phải nguyên nhân |

**Nguyên nhân gốc — nằm ở thiết kế hàm, không phải ở đồ thị.** So sánh hai hàm sinh câu trả lời (cell 3.4):

```python
def answer_flat_rag(question):
    context, retrieved = retrieve_flat_context(question, k=6)   # 6 chunk

def answer_graph_rag(question):
    g = retrieve_graph_context(question, max_hops=2, edge_limit=50, return_debug=True)
    vctx, vdocs = retrieve_flat_context(question, k=4)          # chỉ 4 chunk
```

GraphRAG **đánh đổi 2 chunk vector lấy 8 cạnh đồ thị**. Với câu hỏi này, 8 cạnh đó không chứa NVIDIA, còn chunk bị cắt (hạng 5–6) lại chính là chunk chứa tin NVIDIA. Judge xác nhận: *"completely omits NVIDIA due to its absence in the supplied chunks"*.

Đây không phải "GraphRAG kém hơn", mà là: **khi recall của đồ thị thấp, việc trừ ngân sách vector để nhường chỗ cho đồ thị là một khoản lỗ ròng.**

**Đề xuất khắc phục:**
1. Giữ `k=6` cho cả hai nhánh — subgraph là phần *thêm vào*, không phải phần *thay thế*.
2. Cấp ngân sách động: nếu `collected_edges < 10` thì nâng `k` lên 8 để bù phần đồ thị không đóng góp được.
3. Gốc rễ hơn cả: nâng `EXTRACTION_MAX_CHUNKS` từ 400 lên 1000+ (chi tiết ở Ca 3).

---

## Ca 3 — Cả hai cùng sai: recall của bước NER+RE là nút thắt thật sự

| | |
|---|---|
| **Question ID** | G5000-36 |
| **Nhóm** | cross-doc |
| **Câu hỏi** | *Rows 2532 and 2537 have near-identical Amazon AI headlines. What single event should be stored, and what details can be safely unioned?* |
| **Điểm Flat RAG** | Comp **1** / Faith **5** / Multi-hop **1** |
| **Điểm GraphRAG** | Comp **1** / Faith **1** / Multi-hop **1** |

**Flat RAG** lấy về hai chunk nói về AWS/ADP — đúng là "hai bài Amazon gần giống nhau", nhưng **sai bài**. Judge: *"those chunks describe a different article than the ground truth, indicating a retrieval or alignment failure"* — trung thực với context (Faith 5) nhưng context sai.

**GraphRAG** còn tệ hơn: trả lời về **một cuộc tấn công DDoS ngày 12/10/2023** mà Amazon, Google, Cloudflare cùng chống đỡ. Faith tụt xuống **1** vì nó khẳng định chắc nịch một sự kiện hoàn toàn không liên quan.

**Nguyên nhân gốc:** đây là lỗi hệ thống, không phải lỗi của một câu.

| Số liệu | Giá trị | Ý nghĩa |
|---------|---------|---------|
| Corpus vector (Flat RAG) | **2.635** chunk | Bao phủ toàn bộ 5.000 dòng đầu (2.703 dòng có tin, sau dedup còn 2.635) |
| Chunk gửi qua LLM trích xuất | **400** | Chỉ ~15% corpus được đưa vào đồ thị |
| Số công ty trong 400 chunk đó | **1** | Toàn bộ là chunk của `10Clouds` — xem giải thích bên dưới |
| Đồ thị thu được | 209 node / **124 cạnh** | 1,19 cạnh/node — gần như là các cặp rời rạc |
| Seed entity golden có trong đồ thị | **42/84 = 50%** | Một nửa thực thể được hỏi không tồn tại |
| Câu bị `NO_SEED` | **7/12** | GraphRAG thoái hoá thành vector-only |
| `supernode_events` | **0** | Không node nào đủ lớn để kích hoạt cơ chế cắt tỉa |

Với 124 cạnh trên 209 thực thể, đồ thị **không có đủ đường 2-hop** để một câu multi-hop đi được. `Amazon` có cạnh (degree 7 — cao nhất đồ thị), nhưng cạnh đó trỏ tới sự kiện DDoS chứ không phải tin mở rộng dịch vụ AI.

**Nguyên nhân sâu hơn — 400 chunk được chọn sai chỗ.** Cơ chế chọn chunk ưu tiên theo thứ tự: (1) dòng bằng chứng mà golden dataset trích dẫn, (2) chunk nhắc seed entity của golden dataset, (3) chunk của công ty tần suất cao. Nhưng ở lần chạy này file `graphrag_golden_50_first5000_detailed.csv` **chưa có trong `/content`** khi cell 1.5 và 1.7 chạy, nên hai tầng ưu tiên đầu trả về rỗng:

```
Khong thay /content/graphrag_golden_50_first5000_detailed.csv
Chunk bat buoc (bang chung golden): 0/0
extraction_source: 400 chunk | 1 cong ty | 0 chunk bang chung golden
company
10Clouds    400
```

Pipeline rơi thẳng xuống tầng 3, và trong 5.000 dòng đầu thì `10Clouds` chiếm 1.019 chunk (kế tiếp là `01Synergy` 885, `10Pearls` 708). Kết quả: **cả 400 chunk gửi qua LLM đều thuộc một nguồn cấp tin duy nhất**, được chọn theo thứ tự `chunk_id` — tức là một lát cắt tuỳ tiện, **không chứa dòng 2532/2537** mà câu G5000-36 hỏi tới. Đồ thị không sai; nó chỉ nói về những chuyện khác với những gì bộ câu hỏi hỏi.

**Đề xuất khắc phục, theo thứ tự ưu tiên:**
0. **Đảm bảo hai file golden CSV có mặt trong `/content` TRƯỚC khi chạy cell 1.5.** Đây là sửa chữa rẻ nhất và tác động lớn nhất: nó kích hoạt lại hai tầng ưu tiên chọn chunk, kéo các dòng bằng chứng và các seed entity vào đúng 400 chunk được trích xuất.
1. **Tăng `EXTRACTION_MAX_CHUNKS` 400 → 1000+.** Đây là đòn bẩy lớn nhất. Sau khi chuyển coref/extraction sang chạy song song 4 luồng, chi phí chỉ còn ~5 phút thay vì ~23 phút.
2. **Mở rộng `ALLOWED_RELATIONS`.** Golden dataset khai báo các quan hệ `PROVIDES_ACCESS_TO`, `POWERS`, `CONSIDERING`, `HOSTS_MODEL_FROM`, `PREANNOUNCED` — không nằm trong 8 nhãn mà rubric cho phép. Nhiều câu tin tức không ép được vào 8 nhãn đó nên bị `run_extraction()` loại thẳng, đây là nguồn mất recall có hệ thống.
3. **Chọn chunk theo thực thể mục tiêu** thay vì theo tần suất công ty: ưu tiên chunk có nhắc các seed entity mà bộ câu hỏi sẽ hỏi tới, để các cạnh sinh ra chia sẻ node và tạo hub thay vì cặp rời rạc.

---

## Ca 4 — Judge trả JSON không hợp lệ

| | |
|---|---|
| **Question ID** | G5000-30, G5000-31, G5000-32 |
| **Lỗi** | `RuntimeError: Error code: 400 — Failed to validate JSON. Please adjust your prompt. See 'failed_generation'` |
| **Nhà cung cấp** | Groq · `qwen/qwen3.6-27b` |

**Nguyên nhân gốc:** judge được gọi với `response_format={"type":"json_object"}`. Groq **xác thực JSON ở phía server** và trả HTTP 400 khi model sinh ra JSON hỏng, thay vì trả về chuỗi để client tự parse. Cơ chế retry trong `groq_chat` không cứu được vì đây là lỗi 400 (lỗi yêu cầu) chứ không phải 429/5xx — retry cùng prompt cho ra cùng kết quả.

**Vì sao rơi vào đúng 3 câu này:** đây là các câu có context dài nhất (G5000-30 về Meta trong hai ngữ cảnh AI, G5000-31 yêu cầu sắp xếp 4 mốc thời gian của OpenAI, G5000-32 so sánh plug-in với app-store). Prompt của judge nhúng cả `reference`, `candidate` và `context[:18000]` — khi context dài, model 27B hay sinh JSON có ký tự xuống dòng chưa escape trong trường `rationale`.

**Đề xuất khắc phục:**
1. Bắt riêng lỗi `json_validate_failed`, gọi lại **không** dùng `response_format` rồi tự `parse_json_object()` — pipeline đã có sẵn hàm này với regex bóc ` ```json `.
2. Cắt context truyền cho judge từ 18.000 xuống ~8.000 ký tự — judge chấm độ trung thực, không cần toàn văn.
3. Dùng model judge lớn hơn (`openai/gpt-oss-120b`) cho các câu context dài.

---

## Tổng hợp Failure Mode quan sát được

| Failure mode | Có gặp? | Bằng chứng | Cơ chế phòng vệ trong pipeline |
|--------------|---------|-----------|-------------------------------|
| False coreference → false edge | Không rõ ràng | 152/400 chunk bị coref sửa, 14 chunk có `unresolved_mentions`, 0 batch fail | Conservative prompt; ghi log `unresolved_mentions` thay vì đoán bừa |
| False merge entity | **Không** | `REJECT_GUARD = 0`; lexical guard chưa lần nào phải chặn | Guard `ratio >= 0.72`; cặp nghi ngờ nhất bị chặn bởi ngưỡng cosine (FIS 0.899 < 0.90) |
| Edge thiếu provenance | **Không** | `invalid_provenance_edges = 0` trên 124 cạnh | `bulk_insert_edges()` raise nếu thiếu cột; assert ở cell 2.4 |
| Super-node bùng nổ context | **Không** | `supernode_events = 0` trên toàn bộ 12 câu | Đồ thị quá thưa để kích hoạt; cơ chế cắt tỉa chưa được kiểm chứng trên dữ liệu thật |
| Context tràn token | **Không** | `collected_edges` cao nhất là 16, cách xa `GLOBAL_EDGE_CAP = 250` | `GLOBAL_EDGE_CAP`, `MAX_GRAPH_CONTEXT_CHARS = 14000` |
| LLM extraction trả JSON hỏng | **Không** ở extraction, **có** ở judge | `extraction_errors_df` rỗng; judge lỗi 3/25 | `parse_json_object()` + retry backoff |
| **Recall trích xuất thấp** | **Có — nghiêm trọng nhất** | 50% seed entity vắng mặt; `NO_SEED` 7/12 | *Chưa có cơ chế phòng vệ — đây là khoảng trống lớn nhất của pipeline* |
| **Lỗi im lặng do attribute shadowing** | **Có** | Ca 1 | *Không có; chỉ lộ ra nhờ bảng chẩn đoán `golden_coverage_df`* |

**Kết luận chung:** các failure mode mà bài lab dự đoán trước (super-node, mất provenance, false merge) đều **không** xảy ra — vì cơ chế phòng vệ đã có sẵn, và cũng vì đồ thị quá nhỏ để chúng phát tác. Hai failure mode thực sự gây thiệt hại lại là những cái **không** nằm trong danh sách: recall của bước trích xuất, và một lỗi im lặng do API của pandas.
