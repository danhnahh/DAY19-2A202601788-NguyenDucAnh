# Thuyết Minh Kỹ Thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Nguyễn Đức Anh · **Mã:** 2A202601788
**Khóa học:** AICB-K34 · Track 3: GraphRAG
**Ngày thực hiện:** 19/08/2026

**Cấu hình đã chạy**

| Thành phần | Giá trị |
|---|---|
| Corpus | `hackernoon_subset.csv`, `raw.iloc[:5000]` (khớp `source_scope` của golden dataset) |
| Chunk cho Flat RAG | ~3.000 (`LAB_MAX_CHUNKS`), `CHUNK_WORDS=220`, `overlap=40` |
| Chunk gửi qua LLM trích xuất | **400** (`EXTRACTION_MAX_CHUNKS`) |
| Đồ thị | **209 node · 124 cạnh · `invalid_provenance_edges = 0`** |
| Generator | OpenAI `gpt-4o-mini` |
| Judge | Groq `qwen/qwen3.6-27b` (khác nhà cung cấp với generator) |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2`, cosine qua FAISS `IndexFlatIP` |
| Bộ câu hỏi | 25 câu tự xây (12 multi-hop · 11 cross-doc · 2 factoid) |
| Kết quả chấm được | **12/25** — 13 câu còn lại lỗi, xem `failure_analysis.md` |

---

### 1. Coreference resolution sai ở tình huống nào?

*Nguồn dữ liệu:* `coref_unresolved_df`, `coref_changed_df` (cell 1.7).

**Số liệu thực tế:** 400 chunk · **0 batch thất bại** · **152 chunk (38%) bị coref sửa** · **14 chunk (3,5%) có `unresolved_mentions`**.

Con số đáng chú ý là **14 chunk được cố ý bỏ trống**. Prompt của `resolve_coref_batch()` yêu cầu *"resolve only when the antecedent is clearly supported in the same chunk"*, nên khi mơ hồ, model ghi mention vào `unresolved_mentions` và giữ nguyên văn bản gốc thay vì đoán.

**Vì sao thiết kế bảo thủ là bắt buộc ở đây:** chunk trong lab này chỉ dài trung bình ~45 từ (dataset HackerNoon không có body bài báo, chỉ có `companyName + title + description`). Với ngữ cảnh ngắn như vậy, một đại từ `it`/`the company` thường có **nhiều tiền ngữ ứng viên ngang nhau**. Nếu ép resolve, xác suất gán sai rất cao, và mỗi lần gán sai sẽ tạo ra một **false edge** trong đồ thị — loại lỗi tệ hơn hẳn việc thiếu edge, vì nó khiến GraphRAG trả lời sai một cách tự tin, kèm `source_chunk_id` trông rất đáng tin.

**Ví dụ cụ thể từ `coref_unresolved_df`:**

| chunk_id | Văn bản (rút gọn) | `unresolved_mentions` |
|---|---|---|
| `1c9ba6bec29c51ad659e::c0000` | *"Tech Nation to shut down after the government controversially gives funding to Barclays. 'One of Tech Nati…'"* | `[its]` |
| `09c521d58dd57e637286::c0000` | *"Western Digital's My Cloud is still down but there's a workaround. In the days that followed, the company…"* | `[there's, we didn't hear all that much]` |
| `20b3d7ebcf52c00d7ba2::c0000` | *"Best VoIP Services (July 2023). Toni Matthews-El is a writer and journalist based in Delaware…"* | `[she's]` |
| `0d4b2ff35ba640bdae98::c0000` | *"Head of Information & Support Services. We're here for people with Crohn's and Colitis…"* | `[them]` |

**Phân tích ca đầu tiên — vì sao bỏ trống là quyết định đúng:** trong chunk `1c9ba6be…`, đại từ `its` có **ba tiền ngữ ứng viên** trong cùng một câu: `Tech Nation`, `the government`, và `Barclays`. Không có tín hiệu cú pháp nào trong phạm vi chunk đủ để chọn. Nếu ép resolve và model chọn `Barclays`, ta sẽ có một cạnh gán nguồn tài trợ cho sai tổ chức.

**Hậu quả nếu resolve sai:** cạnh sai vẫn mang `source_chunk_id` và `published_date` hợp lệ nên **qua được toàn bộ sanity check ở cell 2.4** (`invalid_provenance_edges = 0` vẫn đúng). Đây là lý do thiết kế bảo thủ quan trọng hơn tỷ lệ resolve cao: một cạnh thiếu chỉ làm giảm recall, còn một cạnh sai làm GraphRAG trả lời sai **kèm trích dẫn nguồn trông rất đáng tin**.

*(Ghi chú: cột `text` trong output notebook bị cắt ở 120 ký tự do `pd.set_option("display.max_colwidth", 120)`. Muốn xem nguyên văn để đối chiếu `text` → `resolved_text`, chạy `pd.set_option("display.max_colwidth", None)` rồi in lại `coref_changed_df.iloc[0]`.)*

**Hậu quả nếu resolve sai:** giả sử chunk *"AMD announced new AI chips. It will supply them to AWS."* mà `It` bị gán nhầm cho `AWS`, ta sẽ có cạnh `AWS -DEVELOPED-> AI chips` thay vì `AMD -DEVELOPED-> AI chips`. Cạnh sai này vẫn có `source_chunk_id` và `published_date` hợp lệ nên qua được toàn bộ sanity check ở cell 2.4.

---

### 2. Entity resolution threshold là bao nhiêu, vì sao chọn mức đó?

*Nguồn dữ liệu:* `outputs/entity_resolution_audit.csv` — **55 dòng**.

| Tham số | Giá trị | Vai trò |
|---|---|---|
| `MERGE_THRESHOLD` | **0.90** | Ngưỡng cosine để thực sự gộp |
| `AUDIT_THRESHOLD` | **0.60** | Ngưỡng ghi log — thấp hơn nhiều để giữ lại cả các cặp bị từ chối |
| `LEXICAL_GUARD_MIN` | **0.72** | `SequenceMatcher` trên tên đã bỏ hậu tố pháp lý |

**Phân bố quyết định:**

| Decision | Số dòng |
|---|---|
| `REJECT_THRESHOLD` | 48 |
| `MERGE_VECTOR` | 4 |
| `MERGE_MANUAL` | 3 |
| `REJECT_GUARD` | 0 |

**Vì sao 0.90:** đây là quan sát thực tế chứ không phải con số mặc định. Bốn cặp được gộp bằng vector đều nằm sát trên ngưỡng — `Fidelity National Information Services` ↔ `... Inc.` (0.9245), `Fujitsu Limited` ↔ `Fujitsu` (0.9239), `cloud computing service` ↔ `cloud-computing services` (0.9145). Cặp cao nhất **bị từ chối** là `Fidelity National Information Services` ↔ `Fidelity National Information Services (FIS)` ở **0.8992** — chỉ thiếu 0.0008. Đây thực ra là **một false negative**: hai chuỗi này rõ ràng cùng một thực thể (`lexical_ratio = 0.95`).

Bài học: với embedding `all-MiniLM-L6-v2`, khoảng 0.89–0.92 là **vùng xám**, và ngưỡng cứng 0.90 cắt ngay giữa vùng đó. Cách sửa đúng không phải là hạ ngưỡng xuống 0.88 (sẽ kéo theo hàng loạt false merge trong 48 cặp `REJECT_THRESHOLD`), mà là dùng **luật OR**: gộp khi `cosine >= 0.90` **hoặc** khi `cosine >= 0.85 và lexical_ratio >= 0.90`. Cặp FIS sẽ qua được nhánh thứ hai.

**Vai trò của Lexical Guard:** guard tồn tại để chặn trường hợp embedding coi hai công ty **cùng ngành** là giống nhau. Trong lần chạy này guard **chưa lần nào phải chặn** (`REJECT_GUARD = 0`) — không phải vì nó vô dụng, mà vì đồ thị chỉ có 209 thực thể nên chưa xuất hiện cặp nào vừa gần về ngữ nghĩa vừa xa về mặt chữ. Đây là điểm cần trung thực: **cơ chế đã cài nhưng chưa được dữ liệu thật kiểm chứng.**

---

### 3. Candidate nào similarity cao nhưng KHÔNG nên merge?

Bảng audit có **0 dòng `REJECT_GUARD`**, nên không có ca nào bị lexical guard chặn. Ba cặp similarity cao nhất bị từ chối đều rơi vào `REJECT_THRESHOLD`:

Sáu cặp `REJECT_THRESHOLD` có similarity cao nhất:

| Thực thể A | Thực thể B | Cosine | Lexical | Đánh giá của tôi |
|---|---|---|---|---|
| Fidelity National Information Services | Fidelity National Information Services (FIS) | 0.8992 | 0.950 | **False negative** — cùng một thực thể, thiếu 0.0008 |
| L&T Technology Services Ltd. | L&T Technology Services | 0.8661 | **1.000** | **False negative** — lexical bằng 1.0 mà vẫn bị loại |
| Fidelity National Information Services (FIS) | Fidelity National Information Services Inc. | 0.8540 | 0.950 | **False negative** |
| cloud computing unit | cloud computing service | 0.8433 | 0.791 | Loại đúng — "unit" (đơn vị kinh doanh) ≠ "service" (dịch vụ) |
| Cisco | Cisco Systems | 0.8419 | 0.556 | **False negative** — nhưng lexical thấp nên guard cũng sẽ chặn |
| cloud computing service | cloud-computing | 0.8344 | 0.737 | Loại đúng — khái niệm chung vs dịch vụ cụ thể |

**Ba trong sáu cặp đầu bảng là false negative**, và hai trong số đó có `lexical_ratio` ≥ 0.95. Đây là bằng chứng định lượng cho luật OR đã đề xuất ở câu 2: nếu áp `cosine >= 0.85 và lexical >= 0.90`, cả ba cặp FIS/LTTS đều được gộp đúng mà không đụng tới `cloud computing unit` (lexical 0.791) hay `Cisco` (lexical 0.556).

Trường hợp `Cisco` ↔ `Cisco Systems` đáng chú ý riêng: nó là cùng một công ty nhưng `lexical_ratio` chỉ 0.556 vì `SequenceMatcher` phạt nặng khi độ dài chuỗi chênh lệch lớn. Nghĩa là **lexical guard cũng có false negative của riêng nó** — với tên viết tắt phổ biến, cần bổ sung luật "chuỗi này là tiền tố của chuỗi kia sau khi bỏ hậu tố pháp lý".

**Cặp nguy hiểm trên lý thuyết mà bộ dữ liệu này chưa có:** `Amazon` ↔ `Amazon Web Services`. Về ngữ nghĩa cực gần (một là công ty mẹ, một là đơn vị đám mây), nhưng gộp lại sẽ **phá huỷ ngữ nghĩa multi-hop**: câu G5000-27 hỏi phân biệt "AMD cung cấp chip cho nhiều dịch vụ đám mây" với "AWS đang *cân nhắc* chip AI mới của AMD". Nếu `Amazon` và `AWS` là một node, ta không còn phân biệt được phát biểu chung với cam kết cụ thể. Tôi đã chủ động **đưa `aws` vào `MANUAL_ALIASES` ánh xạ sang `Amazon Web Services`** chứ không sang `Amazon`, chính vì lý do này.

---

### 4. Top 3 super-node và degree?

| Hạng | Tên | Type | Degree |
|------|-----|------|--------|
| 1 | **Amazon** | Company | **7** |
| 2 | **Microsoft** | Company | **5** |
| 3 | **Google Cloud** | Company | **4** |

*(Kế tiếp: `cloud computing service` — Technology — 4; `OpenAI`, `AI`, `Google`, `Fidelity National Information Services Inc.`, `Technology` — đều degree 3.)*

**Kết luận quan trọng hơn con số:** degree cao nhất trong toàn đồ thị là **7**, trong khi rubric đặt ngưỡng super-node ở `degree > 100`. Trên toàn bộ 12 câu chấm được, `graph_supernode_events = 0`. Với 209 node và 124 cạnh (**1,19 cạnh/node**), không node nào đạt tới ngưỡng cắt tỉa dù đã hạ ngưỡng.

Ngưỡng gốc của rubric là `degree > 100`. Tôi đã hạ `SUPER_NODE_DEGREE` xuống mức thấp hơn nhiều để cơ chế có cơ hội kích hoạt, nhưng ngay cả vậy vẫn không đủ. **Nói thẳng: cơ chế super-node mitigation trong bài nộp này đã được cài đặt và đã có unit test (`test_supernode_policy()` ở cell 5.1) nhưng chưa được kiểm chứng trên đồ thị thật, vì đồ thị quá thưa.** Muốn kiểm chứng thật cần nâng `EXTRACTION_MAX_CHUNKS` lên hàng nghìn để các công ty lớn tích tụ đủ cạnh.

Công thức cắt tỉa vẫn nguyên vẹn và đúng: `degree > SUPER_NODE_DEGREE` → giới hạn còn `SUPER_NODE_EDGE_CAP` cạnh mới nhất, cộng thêm `GLOBAL_EDGE_CAP = 250` cho toàn bộ lượt duyệt và `MAX_GRAPH_CONTEXT_CHARS = 14000` cho ngữ cảnh sau khi tuyến tính hoá.

---

### 5. Vì sao ưu tiên edge mới nhất có thể đúng/sai?

*Nguồn:* `ORDER BY coalesce(r.published_date,'') DESC LIMIT $limit` trong `recent_edges()` (cell 3.3).

**Ưu điểm:** với tin công nghệ, quan hệ thường bị **ghi đè theo thời gian**. Ví dụ trong chính bộ câu hỏi này, G5000-27 yêu cầu phân biệt bài 01/06 nói chung chung "AMD cấp nguồn cho nhiều dịch vụ đám mây" với bài Reuters 14/06 nói cụ thể "AWS *đang cân nhắc* chip AI mới của AMD". Lấy cạnh mới nhất trước giúp mô hình thấy trạng thái gần hiện tại nhất.

**Rủi ro 1 — mất bối cảnh lịch sử.** Câu G5000-43 hỏi *"cái nào xảy ra trước: thương vụ Axis Security hay thông báo dịch vụ đám mây cho LLM?"* — đây là câu hỏi **về thứ tự thời gian**. Nếu super-node cap cắt mất cạnh cũ hơn, câu hỏi trở nên không trả lời được về mặt nguyên tắc. Điểm multi-hop của GraphRAG ở câu này là **2**, thấp hơn Flat RAG (**3**).

**Rủi ro 2 — `coalesce(...,'')` đẩy cạnh thiếu ngày xuống cuối.** Chuỗi rỗng sắp xếp trước mọi chuỗi ngày theo thứ tự giảm dần, nên cạnh không có `published_date` luôn bị xếp cuối và bị `LIMIT` cắt trước tiên. Trong đồ thị này rủi ro bằng 0 vì `invalid_provenance_edges = 0`, nhưng ở quy mô lớn hơn nơi provenance không hoàn hảo, đây là cơ chế **âm thầm loại bỏ dữ liệu**.

**Rủi ro 3 — mới ≠ đúng.** `published_date` là ngày *xuất bản bài báo*, không phải ngày *xảy ra sự kiện*. Một bài hồi tưởng đăng tháng 12 về sự kiện tháng 3 sẽ được ưu tiên hơn bài tường thuật trực tiếp tháng 3.

---

### 6. Flat RAG thắng ở nhóm câu hỏi nào?

*Nguồn:* `outputs/graphrag_vs_flatrag_summary.csv` (n = 12 câu hợp lệ).

**Flat RAG thắng rõ ở nhóm `multi-hop`** — kết quả đi ngược hoàn toàn giả thuyết của bài lab:

| Metric (multi-hop, n=3) | Flat RAG | GraphRAG | Δ |
|---|---|---|---|
| Comprehensiveness | **5.000** | 4.000 | −1.000 |
| Faithfulness | **5.000** | 4.667 | −0.333 |
| Multi-hop reasoning | **5.000** | 4.333 | −0.667 |

**Lý do — và đây là phần quan trọng nhất của bài thuyết minh:** Flat RAG không thắng vì vector search giỏi hơn suy luận đồ thị. Nó thắng vì **đồ thị quá thưa để đóng góp gì**, trong khi `answer_graph_rag()` vẫn phải trả giá bằng ngân sách ngữ cảnh:

```python
def answer_flat_rag(question):    retrieve_flat_context(question, k=6)
def answer_graph_rag(question):   retrieve_flat_context(question, k=4)   # mất 2 chunk
```

Ở câu G5000-50, đúng 2 chunk bị cắt đó chứa tin về NVIDIA, và GraphRAG trả lời *"the context does not provide specific information about NVIDIA"* trong khi Flat RAG trả lời đủ cả ba công ty. Chi tiết ở `failure_analysis.md` Ca 2.

Nói cách khác: với đồ thị 124 cạnh, **hybrid retrieval là một phép trừ chứ không phải phép cộng.**

---

### 7. GraphRAG thắng ở nhóm nào?

| Metric (cross-doc, n=7) | Flat RAG | GraphRAG | Δ |
|---|---|---|---|
| Comprehensiveness | 3.143 | 3.143 | 0.000 |
| Faithfulness | 4.571 | 4.286 | −0.286 |
| **Multi-hop reasoning** | 3.429 | **3.571** | **+0.143** |

Nhóm `cross-doc` là nơi duy nhất GraphRAG nhỉnh hơn, và chỉ ở đúng một metric. Ca thắng thuyết phục nhất là **G5000-45**:

| | Flat RAG | GraphRAG |
|---|---|---|
| Comprehensiveness | 2 | **3** |
| Faithfulness | 2 | **4** |
| Multi-hop reasoning | 2 | **4** |

Câu hỏi: *"Hai dòng 261 và 891 đều mô tả L&T Technology Services và Qualcomm được Thales chọn. Đồ thị production nên tránh đếm trùng thế nào?"*

- **Flat RAG** đề xuất tạo một node ghép `"L&T Technology Services and Qualcomm Partnership with Thales"` — một cấu trúc bịa ra, sai về mô hình dữ liệu.
- **GraphRAG** trả lời đúng: gộp thành **một cạnh duy nhất** kèm ngày `2023-02-23`, không nhân bản theo số nguồn.

**Vì sao GraphRAG đúng ở đây:** `matched_seeds = 2` (cả LTTS và Qualcomm đều có trong đồ thị) và `collected_edges = 2`. Ngữ cảnh đồ thị được tuyến tính hoá **chính là một ví dụ mẫu** về cách biểu diễn quan hệ: `A -PARTNERED_WITH-> B | date=... | chunk=...`. Mô hình nhìn thấy cấu trúc đó và bắt chước. Đây là giá trị thật của GraphRAG mà bảng điểm trung bình che mất: **nó áp đặt một lược đồ, và lược đồ đó dẫn dắt câu trả lời.**

**Điều kiện để GraphRAG thắng, rút ra từ dữ liệu:** trong 12 câu chấm được, có **7 câu bị `NO_SEED`** — GraphRAG thoái hoá thành vector-only. GraphRAG chỉ thắng ở những câu mà cả hai thực thể trong câu hỏi đều tồn tại trong đồ thị (`matched_seeds >= 2`).

---

### 8. Trade-off latency / token?

| Metric (ALL, n=12) | Flat RAG | GraphRAG | Δ | Tỷ lệ |
|--------|----------|----------|---|---|
| Latency (s) | 2.278 | **2.081** | −0.197 | ×0.91 |
| Token usage | 809.75 | 823.92 | +14.17 | ×1.02 |

**Kết quả thoạt nhìn có vẻ nghịch lý: GraphRAG *nhanh hơn*.** Giải thích nằm ở phân rã theo nhóm:

| Nhóm | Latency Δ | Token Δ | Diễn giải |
|---|---|---|---|
| factoid (n=2) | −0.324 (×0.79) | **−201.5** (×0.75) | Cả hai câu đều `NO_SEED` → context đồ thị rỗng, chỉ còn 4 chunk vector thay vì 6 |
| cross-doc (n=7) | −0.475 (×0.81) | −12.6 (×0.98) | 5/7 câu `NO_SEED` → tương tự |
| **multi-hop (n=3)** | **+0.536 (×1.25)** | **+220.3 (×1.26)** | Đồ thị thực sự trả về cạnh → đây mới là chi phí thật |

Con số tổng thể bị **bóp méo bởi 7 câu `NO_SEED`**: khi đồ thị không trả về gì, GraphRAG rẻ hơn đơn giản vì nó dùng ít chunk vector hơn. **Chi phí thật của GraphRAG là dòng multi-hop: đắt hơn ~25% cả về thời gian lẫn token.**

**Chi phí ẩn không nằm trong bảng này:**
- Mỗi truy vấn GraphRAG tốn thêm **1 lời gọi LLM** cho `extract_seeds()`, cộng **N truy vấn Cypher** cho `node_degree()` và `recent_edges()` — các Cypher này chạy trên Neo4j Aura qua mạng.
- **Chi phí index một lần** (không tính vào latency truy vấn): 400 chunk × 2 lượt LLM (coref + NER/RE) = 80 lời gọi API. Chạy tuần tự mất ~9 phút chỉ riêng coref; sau khi chuyển sang `ThreadPoolExecutor` 4 luồng còn **1 phút 54 giây**. Flat RAG chỉ tốn chi phí embedding cục bộ cho 3.000 chunk (~30 giây, không tốn tiền API).

**Kết luận về trade-off:** ở quy mô đồ thị này, GraphRAG **đắt hơn 25% khi hoạt động và không cho lợi ích đo được**. Đó là kết luận đúng cho *bộ dữ liệu này*, không phải cho kiến trúc GraphRAG nói chung — điều kiện hoà vốn là đồ thị phải đủ đặc để `NO_SEED` xuống gần 0.

---

### 9. AI Coding Agent đề xuất gì mà tôi KHÔNG dùng, vì sao?

**Đề xuất bị từ chối #1 — Mở rộng `ALLOWED_RELATIONS` cho khớp golden dataset.**

Bộ golden dataset tôi tự xây khai báo các quan hệ `PROVIDES_ACCESS_TO`, `POWERS`, `CONSIDERING`, `HOSTS_MODEL_FROM`, `PREANNOUNCED` trong cột `required_relations`. Agent chỉ ra rằng đồ thị không bao giờ lưu được các nhãn đó, và đề xuất bổ sung chúng vào allowlist để tăng recall.

**Lý do từ chối:** rubric quy định cứng 8 quan hệ (`ACQUIRED`, `DEVELOPED`, `INVESTED_IN`, `FOUNDED`, `WORKED_AT`, `PARTNERED_WITH`, `USES`, `LEADS`). Đổi schema sẽ được điểm benchmark cao hơn nhưng vi phạm đặc tả bài. Tôi giữ nguyên allowlist và **ghi nhận đây là nguồn mất recall có hệ thống** trong `failure_analysis.md`, để người chấm thấy tôi hiểu vấn đề chứ không phải bỏ sót.

**Đề xuất bị từ chối #2 — Hạ ngưỡng entity resolution xuống 0.88 để cứu cặp FIS.**

Agent phát hiện cặp `Fidelity National Information Services` ↔ `... (FIS)` bị từ chối ở 0.8992 và đề xuất hạ ngưỡng.

**Lý do từ chối:** bảng audit có **48 cặp** nằm dưới 0.90. Hạ ngưỡng để cứu 1 cặp sẽ kéo theo cả một vùng chưa được kiểm tra. Giải pháp đúng là luật OR hai chiều (`cosine >= 0.90` **hoặc** `cosine >= 0.85 và lexical >= 0.90`) — sửa đúng ca cần sửa mà không nới lỏng toàn hệ thống. Đây chính là lý do tôi hạ `AUDIT_THRESHOLD` xuống 0.60: **để có dữ liệu mà quyết định, thay vì đoán.**

**Đề xuất bị từ chối #3 — Dùng lại kết quả từ checkpoint cho các câu lỗi.**

Cơ chế checkpoint ban đầu resume mọi dòng đã ghi, kể cả dòng có `error`. Sau khi sửa bug ở Ca 1, cách làm đó sẽ khiến 10 câu lỗi **được kế thừa lại nguyên lỗi cũ**. Tôi sửa để chỉ resume các dòng `error == ""`.

---

### 10. Scale lên 350MB (~100.000 bài): bottleneck đầu tiên là gì?

**Bottleneck #1 — Chi phí và thời gian gọi LLM cho bước trích xuất.** Đây là nút thắt áp đảo, không phải Neo4j.

Đo thực tế: 400 chunk cần 80 lời gọi LLM (40 coref + 40 NER/RE), tuần tự mất ~18 phút. Ngoại suy tuyến tính lên 100.000 bài (~100.000 chunk): **20.000 lời gọi ≈ 75 giờ tuần tự**. Với 4 luồng như hiện tại vẫn còn ~19 giờ.

*Giải pháp:*
1. **Batch API bất đồng bộ** (OpenAI Batch API) — rẻ hơn ~50% và không bị rate limit theo phút, đổi lấy độ trễ tính bằng giờ. Trích xuất là tác vụ offline nên đánh đổi này hoàn toàn chấp nhận được.
2. **Lọc trước bằng NER cục bộ**: chạy spaCy/GLiNER để loại bỏ chunk không chứa thực thể nào thuộc 3 loại cho phép, chỉ gửi phần còn lại qua LLM. Trong lab này 400 chunk chỉ sinh 124 cạnh — tỷ lệ chunk "vô ích" rất cao.
3. **Chốt chặn idempotent**: lưu cache theo `sha1(chunk_text)` để chạy lại không phải trả tiền lần hai.

**Bottleneck #2 — Entity resolution có độ phức tạp bậc hai.**

`build_resolution_map()` hiện tự tra cứu top-k trên toàn bộ ma trận với `faiss.IndexFlatIP`, tức là **quét toàn bộ**. Với 209 thực thể thì không thành vấn đề; với ~500.000 thực thể mention thì `IndexFlatIP` là O(n²) về tổng chi phí và ma trận embedding không còn vừa RAM.

*Giải pháp:*
1. Đổi sang **`IndexIVFFlat`** hoặc **HNSW** — tra cứu xấp xỉ, giảm từ O(n) xuống ~O(log n) mỗi truy vấn, đổi lấy sai số recall nhỏ (chấp nhận được vì đã có lexical guard chặn hậu kiểm).
2. **Chia khối trước khi so khớp** (blocking): chỉ so các mention trong cùng `entity_type` **và** cùng ký tự đầu đã chuẩn hoá, cắt bỏ phần lớn cặp không bao giờ khớp.
3. Thay `UF` (union-find) trong bộ nhớ bằng **Neo4j GDS Weakly Connected Components**, để việc gom cụm chạy ngay trong database thay vì kéo hết ra Python.

**Bottleneck #3 — Super-node xuất hiện thật ở quy mô này.**

Ở 400 chunk, `supernode_events = 0`. Ở 100.000 bài, các thực thể như `AI`, `Microsoft`, `cloud computing` gần như chắc chắn vượt degree 1.000+. Lúc đó `MAX_GRAPH_CONTEXT_CHARS = 14000` và `GLOBAL_EDGE_CAP = 250` mới thực sự là ràng buộc ràng buộc — và chiến lược "lấy cạnh mới nhất" sẽ lộ nhược điểm ở Q5.

*Giải pháp:* chuyển từ cắt tỉa theo *độ mới* sang cắt tỉa theo *độ liên quan* — xếp hạng cạnh theo `cosine(embedding(evidence), embedding(query))` rồi mới cắt, thay vì `ORDER BY published_date DESC`.
