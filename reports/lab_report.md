# Báo Cáo Thực Hành & Thuyết Minh Kỹ Thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Nguyễn Đức Anh · **Mã học viên:** 2A202601788
**Khóa học:** AICB-K34 · Track 3: GraphRAG
**Ngày thực hiện:** 19/08/2026

**Tài liệu liên quan:** [`technical_defense.md`](technical_defense.md) (10 câu thuyết minh) · [`failure_analysis.md`](failure_analysis.md) (4 ca lỗi) · [`reflection_NguyenDucAnh.md`](reflection_NguyenDucAnh.md)
**Dữ liệu:** [`../outputs/graphrag_eval_results.csv`](../outputs/graphrag_eval_results.csv) · [`../outputs/graphrag_vs_flatrag_summary.csv`](../outputs/graphrag_vs_flatrag_summary.csv) · [`../outputs/entity_resolution_audit.csv`](../outputs/entity_resolution_audit.csv) · [`../outputs/golden_coverage.csv`](../outputs/golden_coverage.csv)

---

## Tóm tắt điều hành

Pipeline chạy hết 5 module và dựng được đồ thị **209 node · 124 cạnh · 0 cạnh thiếu provenance**. Trên 25 câu hỏi golden, **12 câu cho điểm hợp lệ**; 13 câu còn lại trả về exception và được phân tích như ca lỗi thay vì bị loại bỏ.

**Kết luận chính, đi ngược giả thuyết của bài lab: GraphRAG thua Flat RAG ở nhóm multi-hop (4.00 vs 5.00 comprehensiveness).** Nguyên nhân đã truy được đến gốc và **không** nằm ở kiến trúc retrieval mà ở hai chỗ cụ thể:

1. **Recall của bước trích xuất quá thấp, và 400 chunk lại chọn sai chỗ.** Chỉ 400/2.635 chunk (~15% corpus) được gửi qua LLM, cho ra 1,19 cạnh/node. Nặng hơn: hai file golden CSV chưa có trong `/content` khi cell 1.5–1.7 chạy, nên cơ chế ưu tiên chunk theo bằng chứng và seed entity không kích hoạt được, và **cả 400 chunk rơi vào một nguồn cấp tin duy nhất (`10Clouds`)**. Kết quả: **50% (42/84) seed entity mà bộ câu hỏi cần không tồn tại trong đồ thị**, và **7/12 câu bị `NO_SEED`**.
2. **Hybrid retrieval là phép trừ, không phải phép cộng.** `answer_graph_rag()` dùng `k=4` chunk vector trong khi `answer_flat_rag()` dùng `k=6`. Khi đồ thị không đóng góp gì, GraphRAG mất trắng 2 chunk.

**Một cảnh báo về chính bảng số này:** 13/25 câu trả về exception, và 10 trong số đó do một bug làm mất dòng gán biến ở nhánh fuzzy matching. Vì bug chỉ nổ khi fuzzy match **thành công**, 10 câu bị loại chính là 10 câu GraphRAG có seed thật. Bảng benchmark n=12 vì vậy **đánh giá thấp GraphRAG một cách có hệ thống** — chi tiết ở [`failure_analysis.md`](failure_analysis.md) Ca 1.

Đây là kết luận về **bộ dữ liệu này ở quy mô này**, không phải về GraphRAG nói chung. Điều kiện hoà vốn đã định lượng được: tỷ lệ `NO_SEED` phải tiến gần 0.

---

## 📌 PHẦN 1: THUYẾT MINH KỸ THUẬT & PHÂN TÍCH CA LỖI

### 1. Coreference Resolution (Phân giải đại từ)

**Số liệu thực tế (cell 1.7):** 400 chunk · **0 batch thất bại** · **152 chunk (38%) bị sửa** · **14 chunk (3,5%) có `unresolved_mentions`**.

- **Ví dụ từ dữ liệu** (`coref_unresolved_df`):

| chunk_id | Văn bản (rút gọn) | `unresolved_mentions` |
|---|---|---|
| `1c9ba6bec29c51ad659e::c0000` | *"Tech Nation to shut down after the government controversially gives funding to Barclays…"* | `[its]` |
| `09c521d58dd57e637286::c0000` | *"Western Digital's My Cloud is still down… In the days that followed, the company…"* | `[there's, we didn't hear…]` |
| `20b3d7ebcf52c00d7ba2::c0000` | *"Best VoIP Services (July 2023). Toni Matthews-El is a writer…"* | `[she's]` |

- **Hiện tượng:** con số đáng chú ý không phải 152 chunk được sửa mà là **14 chunk được cố ý bỏ trống**. Chunk trong dataset này chỉ dài trung bình **43,2 từ** (HackerNoon dump không có body bài báo, chỉ có `companyName + title + description`), nên một đại từ `it` / `its` / `the company` thường có **nhiều tiền ngữ ứng viên ngang nhau**. Trong chunk `1c9ba6be…`, đại từ `its` có **ba ứng viên trong cùng một câu**: `Tech Nation`, `the government`, `Barclays` — không tín hiệu cú pháp nào trong phạm vi chunk đủ để chọn. Prompt bảo thủ ghi vào `unresolved_mentions` thay vì đoán.
- **Hậu quả đối với Graph nếu resolve sai:** với chunk *"AMD announced new AI chips. It will supply them to AWS."*, nếu `It` bị gán nhầm cho `AWS` ta sẽ có cạnh `AWS -DEVELOPED-> AI chips` thay vì `AMD -DEVELOPED-> AI chips`. Cạnh sai này vẫn mang `source_chunk_id` và `published_date` hợp lệ nên **qua được toàn bộ sanity check ở cell 2.4** — đây là lý do thiết kế bảo thủ quan trọng hơn tỷ lệ resolve cao.

---

### 2. Entity Resolution Threshold & Lexical Guard

**Cấu hình:** `MERGE_THRESHOLD = 0.90` · `AUDIT_THRESHOLD = 0.60` · `LEXICAL_GUARD_MIN = 0.72`
**Bảng audit:** **55 dòng** (`outputs/entity_resolution_audit.csv`)

| Decision | Số dòng |
|---|---|
| `REJECT_THRESHOLD` | 48 |
| `MERGE_VECTOR` | 4 |
| `MERGE_MANUAL` | 3 |
| `REJECT_GUARD` | **0** |

**Cặp bị Lexical Guard chặn:** **không có ca nào.** Đây là điểm phải báo cáo trung thực — guard đã được cài đặt và hoạt động, nhưng với đồ thị chỉ 209 thực thể, chưa xuất hiện cặp nào vừa gần về ngữ nghĩa vừa xa về mặt chữ. **Cơ chế chưa được dữ liệu thật kiểm chứng.**

**Phát hiện đắt giá hơn — một false negative ngay sát ngưỡng:**

| Thực thể A | Thực thể B | Cosine | Lexical | Quyết định |
|---|---|---|---|---|
| Fidelity National Information Services | Fidelity National Information Services (FIS) | **0.8992** | 0.950 | `REJECT_THRESHOLD` |
| Fidelity National Information Services | Fidelity National Information Services Inc. | 0.9245 | 1.000 | `MERGE_VECTOR` |
| Fujitsu Limited | Fujitsu | 0.9239 | 1.000 | `MERGE_VECTOR` |
| cloud computing service | cloud-computing services | 0.9145 | 0.936 | `MERGE_VECTOR` |

Cặp FIS rõ ràng là **cùng một thực thể** nhưng bị từ chối vì thiếu đúng **0.0008**. Với embedding `all-MiniLM-L6-v2`, khoảng **0.89–0.92 là vùng xám** và ngưỡng cứng 0.90 cắt ngay giữa vùng đó.

**Cách sửa đúng không phải hạ ngưỡng** (48 cặp `REJECT_THRESHOLD` sẽ ùa vào), mà là **luật OR hai chiều**: gộp khi `cosine >= 0.90` **hoặc** (`cosine >= 0.85` **và** `lexical_ratio >= 0.90`). Chính vì cần dữ liệu để ra quyết định này mà `AUDIT_THRESHOLD` được hạ xuống 0.60 — ghi log cả các cặp bị từ chối thay vì chỉ ghi cặp được gộp.

**Một quyết định chủ động về schema:** tôi ánh xạ `aws` → `Amazon Web Services` chứ **không** gộp vào `Amazon`. Về ngữ nghĩa hai thực thể rất gần, nhưng gộp lại sẽ phá huỷ khả năng trả lời câu G5000-27 (phân biệt "AMD cấp nguồn cho nhiều dịch vụ đám mây" với "AWS *đang cân nhắc* chip AI mới của AMD").

---

### 3. Đồ thị & Super-node Mitigation

**Đặc trưng đồ thị:** 209 node · 124 cạnh · **1,19 cạnh/node** · `invalid_provenance_edges = 0`

- **Top 3 Super-nodes:**

| Hạng | Tên thực thể | Loại thực thể | Bậc kết nối |
|------|--------------|---------------|-------------|
| 1 | **Amazon** | Company | **7** |
| 2 | **Microsoft** | Company | **5** |
| 3 | **Google Cloud** | Company | **4** |

**Kết luận quan trọng hơn ba con số trên:** degree cao nhất toàn đồ thị là **7**, trong khi rubric đặt ngưỡng super-node ở `degree > 100` — chênh nhau hơn một bậc độ lớn. Trên toàn bộ 12 câu chấm được, **`graph_supernode_events = 0`** và `collected_edges` cao nhất chỉ là **16**, cách rất xa `GLOBAL_EDGE_CAP = 250`. Tôi đã hạ `SUPER_NODE_DEGREE` xuống thấp hơn nhiều để cơ chế có cơ hội kích hoạt nhưng vẫn không đủ.

**Nói thẳng: super-node mitigation trong bài nộp này đã được cài đặt đầy đủ và có unit test (`test_supernode_policy()` ở cell 5.1), nhưng chưa được kiểm chứng trên đồ thị thật vì đồ thị quá thưa.**

**Ưu điểm & Rủi ro của Temporal Mitigation** (`ORDER BY coalesce(r.published_date,'') DESC LIMIT $limit`):

- *Ưu điểm:* tin công nghệ thường bị **ghi đè theo thời gian**. Chính bộ câu hỏi này có G5000-27 yêu cầu phân biệt bài 01/06 nói chung chung với bài Reuters 14/06 nói cụ thể — lấy cạnh mới nhất giúp mô hình thấy trạng thái gần hiện tại nhất.
- *Rủi ro 1 — mất bối cảnh lịch sử:* câu G5000-43 hỏi **thứ tự thời gian** ("thương vụ Axis Security hay thông báo dịch vụ LLM xảy ra trước?"). Điểm multi-hop của GraphRAG ở câu này là **2**, thấp hơn Flat RAG (**3**). Với dạng câu hỏi này, cạnh cũ *chính là* câu trả lời.
- *Rủi ro 2 — `coalesce(...,'')` âm thầm loại dữ liệu:* chuỗi rỗng sắp trước mọi chuỗi ngày khi sắp giảm dần, nên cạnh thiếu `published_date` luôn bị `LIMIT` cắt trước tiên. Ở đồ thị này rủi ro bằng 0 (`invalid_provenance_edges = 0`), nhưng ở quy mô lớn hơn thì đây là cơ chế mất dữ liệu không có cảnh báo.
- *Rủi ro 3 — mới ≠ đúng:* `published_date` là ngày **xuất bản bài báo**, không phải ngày **xảy ra sự kiện**.

---

### 4. So sánh Thực nghiệm (Flat RAG vs GraphRAG)

#### Bảng tổng hợp Benchmark — toàn bộ (n = 12 câu hợp lệ)

| Tiêu chí đánh giá | Flat RAG | GraphRAG | Δ | Nhận xét phân tích |
|-------------------|----------|----------|---|-------------------|
| **Comprehensiveness (1–5)** | **3.917** | 3.667 | −0.250 | Hai bên gần nhau, nhưng con số tổng bị 7 câu `NO_SEED` làm phẳng |
| **Faithfulness (1–5)** | **4.750** | 4.500 | −0.250 | GraphRAG thấp hơn do 1 ca hallucination nặng (G5000-36, Faith 1/5) |
| **Multi-hop Reasoning (1–5)** | **4.083** | 4.000 | −0.083 | Chênh lệch không đáng kể ở mức tổng |
| **Latency trung bình (s)** | 2.278 | **2.081** | −0.197 (×0.91) | GraphRAG *có vẻ* nhanh hơn — xem giải thích bên dưới |
| **Token usage trung bình** | **809.75** | 823.92 | +14.17 (×1.02) | Gần như hoà |

#### Phân rã theo nhóm — đây mới là bức tranh thật

| Nhóm | n | Comp (F→G) | Faith (F→G) | Multi-hop (F→G) | Latency Δ | Token Δ |
|---|---|---|---|---|---|---|
| **factoid** | 2 | 5.00 → 5.00 | 5.00 → 5.00 | 5.00 → 5.00 | −0.32 (×0.79) | **−201.5** (×0.75) |
| **cross-doc** | 7 | 3.14 → 3.14 | 4.57 → 4.29 | 3.43 → **3.57** | −0.48 (×0.81) | −12.6 (×0.98) |
| **multi-hop** | 3 | **5.00 → 4.00** | 5.00 → 4.67 | **5.00 → 4.33** | **+0.54 (×1.25)** | **+220.3 (×1.26)** |

**Ba điều bảng này nói ra:**

1. **Con số latency/token tổng thể bị bóp méo bởi 7 câu `NO_SEED`.** Khi đồ thị không trả về gì, GraphRAG rẻ hơn đơn giản vì nó dùng 4 chunk vector thay vì 6. **Chi phí thật của GraphRAG là dòng multi-hop: đắt hơn ~25% cả thời gian lẫn token.**
2. **Nhóm cross-doc là nơi duy nhất GraphRAG nhỉnh hơn** (+0.143 ở multi-hop reasoning), và chỉ ở đúng một metric.
3. **Nhóm factoid hoà tuyệt đối 5/5/5** — cả hai đều trả lời đúng, GraphRAG rẻ hơn 25% token vì cả 2 câu đều `NO_SEED` nên context ngắn hơn. Đây là bằng chứng cho luận điểm kinh điển: **câu factoid không cần đồ thị.**

#### Phân tích 2 Ca lỗi Điển hình

*(Bốn ca đầy đủ ở [`failure_analysis.md`](failure_analysis.md). Dưới đây là hai ca cốt lõi.)*

**1. Ca GraphRAG thành công, Flat RAG thất bại — G5000-45** (cross-doc)

> *"Rows 261 and 891 describe L&T Technology Services and Qualcomm being selected by Thales. How should a production graph avoid double-counting this?"*

| | Flat RAG | GraphRAG |
|---|---|---|
| Comprehensiveness | 2 | **3** |
| Faithfulness | 2 | **4** |
| Multi-hop reasoning | 2 | **4** |

- *Tại sao Flat RAG thất bại?* Nó đề xuất tạo một node ghép `"L&T Technology Services and Qualcomm Partnership with Thales"` — một cấu trúc bịa ra, sai về mô hình dữ liệu. Judge: *"proposes an unconventional graph structure instead of canonicalizing the event"*.
- *GraphRAG đã giải quyết như thế nào?* `matched_seeds = 2` (cả LTTS lẫn Qualcomm đều có trong đồ thị), `collected_edges = 2`. Ngữ cảnh đồ thị sau khi tuyến tính hoá **chính là một ví dụ mẫu** về cách biểu diễn quan hệ: `A -PARTNERED_WITH-> B | date=2023-02-23 | chunk=...`. Mô hình nhìn thấy lược đồ đó và bắt chước, trả lời đúng là gộp thành **một cạnh duy nhất** với nhiều provenance.
- *Bài học:* giá trị thật của GraphRAG ở đây không phải là cấp thêm dữ kiện, mà là **áp đặt một lược đồ dẫn dắt câu trả lời** — thứ mà bảng điểm trung bình che mất.

**2. Ca GraphRAG thất bại — G5000-50** (multi-hop)

> *"Compare the chip-related AI positioning of NVIDIA, AMD, and Intel. What distinct signal is reported for each?"*

| | Flat RAG | GraphRAG |
|---|---|---|
| Comprehensiveness | **5** | 2 |
| Faithfulness | **5** | 4 |
| Multi-hop reasoning | **5** | 3 |

- *Nguyên nhân:* GraphRAG trả lời *"**NVIDIA**: The context does not provide specific information about NVIDIA's AI positioning."* Loại trừ từng mắt xích bằng `graph_debug["diagnostics"]`: không phải super-node cap (`supernode_events = 0`), không phải `GLOBAL_EDGE_CAP` (`collected_edges = 8` ≪ 250), không phải tràn ký tự. Nguyên nhân gốc nằm ở **thiết kế hàm**:

```python
def answer_flat_rag(question):    retrieve_flat_context(question, k=6)   # 6 chunk
def answer_graph_rag(question):   retrieve_flat_context(question, k=4)   # chỉ 4 chunk
```

  GraphRAG **đánh đổi 2 chunk vector lấy 8 cạnh đồ thị**, và 8 cạnh đó không chứa NVIDIA trong khi 2 chunk bị cắt thì có. Judge xác nhận: *"completely omits NVIDIA due to its absence in the supplied chunks"*.
- *Đề xuất khắc phục:* (a) giữ `k=6` cho cả hai nhánh — subgraph phải là phần **thêm vào**, không phải phần **thay thế**; (b) cấp ngân sách động: nếu `collected_edges < 10` thì nâng `k` lên 8; (c) gốc rễ hơn cả là nâng `EXTRACTION_MAX_CHUNKS` 400 → 1000+.

---

### 5. Đánh đổi (Trade-offs) & Kiểm soát AI Coding Agent

**Đánh đổi Quality vs Cost vs Latency:**

Ở quy mô đồ thị này, GraphRAG **đắt hơn 25% khi thực sự hoạt động và không cho lợi ích chất lượng đo được**. Chi phí ẩn không nằm trong bảng benchmark:

- Mỗi truy vấn GraphRAG tốn thêm **1 lời gọi LLM** (`extract_seeds()`) cộng **N truy vấn Cypher** (`node_degree()`, `recent_edges()`) chạy qua mạng tới Neo4j Aura.
- **Chi phí index một lần:** 400 chunk × 2 lượt LLM = 80 lời gọi API. Chạy tuần tự, riêng coref đã mất **9 phút 05**; sau khi chuyển sang `ThreadPoolExecutor` 4 luồng còn **1 phút 54**. Flat RAG chỉ tốn embedding cục bộ cho 3.000 chunk (~30 giây, không tốn tiền API).

Điều kiện hoà vốn đã định lượng được: **tỷ lệ `NO_SEED` phải tiến gần 0**. Ở mức 7/12 như hiện tại, đầu tư vào đồ thị là lỗ ròng.

**Quyết định từ chối AI Coding Agent:**

1. **Từ chối mở rộng `ALLOWED_RELATIONS`.** Golden dataset khai báo `PROVIDES_ACCESS_TO`, `POWERS`, `CONSIDERING`, `HOSTS_MODEL_FROM`, `PREANNOUNCED` — thêm vào sẽ tăng recall và điểm benchmark, nhưng vi phạm đặc tả 8 quan hệ của rubric. Giữ nguyên và ghi nhận đây là nguồn mất recall có hệ thống.
2. **Từ chối hạ ngưỡng ER xuống 0.88** để cứu cặp FIS. Có tới **48 cặp** nằm dưới 0.90; nới ngưỡng để cứu 1 cặp là mở cửa cho cả một vùng chưa kiểm tra. Giải pháp đúng là luật OR hai chiều.
3. **Từ chối bản vá che triệu chứng.** Khi `", ".join(s["name"] ...)` nổ, agent gợi ý bọc `str(... or "?")`. Cách đó chạy được nhưng sẽ biến mọi tên thành `"?"` và **giấu luôn bug gốc** (`pandas.Series.name` che mất cột `name`). Phải sửa thành `r["name"]` mới là sửa thật.

**Giải pháp scale 350MB (~100.000 bài):**

- **Bottleneck #1 — chi phí gọi LLM cho trích xuất**, không phải Neo4j. Ngoại suy từ đo thực tế: 400 chunk cần 80 lời gọi ≈ 18 phút tuần tự → 100.000 chunk cần **20.000 lời gọi ≈ 75 giờ tuần tự** (~19 giờ với 4 luồng). *Giải pháp:* Batch API bất đồng bộ (rẻ hơn ~50%, không bị rate limit theo phút, độ trễ tính bằng giờ — chấp nhận được vì trích xuất là tác vụ offline); lọc trước bằng NER cục bộ (spaCy/GLiNER) để loại chunk không chứa thực thể thuộc 3 loại cho phép; cache idempotent theo `sha1(chunk_text)`.
- **Bottleneck #2 — Entity Resolution bậc hai.** `faiss.IndexFlatIP` là quét toàn bộ; với ~500.000 mention thì ma trận embedding không vừa RAM. *Giải pháp:* đổi sang `IndexIVFFlat`/HNSW; **blocking** theo `entity_type` + ký tự đầu đã chuẩn hoá; thay union-find trong bộ nhớ bằng **Neo4j GDS Weakly Connected Components**.
- **Bottleneck #3 — super-node xuất hiện thật.** Ở quy mô này, `AI` / `Microsoft` / `cloud computing` gần như chắc chắn vượt degree 1.000+. *Giải pháp:* chuyển từ cắt tỉa theo **độ mới** sang cắt tỉa theo **độ liên quan** — xếp hạng cạnh bằng `cosine(embedding(evidence), embedding(query))` rồi mới `LIMIT`.

---

### 6. Bonus — Community Detection & Global Search

**Phân cụm (NetworkX `greedy_modularity_communities`):** **87 cộng đồng** trên đồ thị 209 node / 124 cạnh.

| Cụm | Số thành viên |
|---|---|
| 0 | 12 |
| 1 | 8 |
| 2 | 5 |
| 3–5 | 4 |
| 6–9 | 3 |

**Đọc con số này:** 87 cụm cho 209 node nghĩa là kích thước cụm trung bình chỉ **~2,4 node**, và cụm lớn nhất cũng chỉ có 12 thành viên. Đây là **xác nhận độc lập** cho kết luận ở mục 3: với 1,19 cạnh/node, đồ thị không có cấu trúc cộng đồng thật — nó là một tập hợp các cặp và bộ ba rời rạc. Modularity clustering không tạo ra được cấu trúc mà dữ liệu không có.

Điều này có ý nghĩa trực tiếp với Global Search: kỹ thuật này chỉ phát huy khi mỗi cụm đủ lớn để bản tóm tắt của nó mang thông tin cấp chủ đề. Với cụm 2–3 node, "bản tóm tắt cộng đồng" gần như chỉ là chép lại một cạnh. Vì vậy tôi chỉ sinh báo cáo cho các cụm có **≥ 3 thành viên**, lấy 8 cụm lớn nhất.

`community_id`, `community_title` và `community_summary` đều được nạp ngược vào Neo4j bằng `UNWIND` theo batch, để truy vấn vĩ mô sau này dùng lại được mà không phải phân cụm lại.

**Kết quả Global Search vs Flat RAG trên câu hỏi vĩ mô:** ⟨ĐIỀN sau khi chạy cell `Bonus — Community Reports + Global Search`: chép bảng so sánh latency / token / độ bao phủ ở cuối output⟩

---

## 📌 PHẦN 2: SUY NGẪM & KẾ HOẠCH ĐỒ ÁN

*(Bản đầy đủ ở [`reflection_NguyenDucAnh.md`](reflection_NguyenDucAnh.md). Dưới đây là phần cốt lõi.)*

### 1. Mapping Bài giảng vào Code

| Khái niệm trong bài giảng | Module | Hàm / Khối code | Quan sát thực tế & Đánh giá |
|--------------------------|--------|-----------------|-----------------------------|
| **Conservative Coreference** | M1 | `resolve_coref_batch()` | 152/400 chunk bị sửa, **14 chunk cố ý bỏ trống**. Chunk ~45 từ nên đại từ thường đa nghĩa; bỏ trống an toàn hơn đoán sai vì false edge vẫn qua được mọi sanity check. |
| **Schema & Allowlist Guard** | M2 | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS` | `extraction_errors_df` rỗng, mọi cạnh hợp lệ. Nhưng guard **vừa là lá chắn vừa là nút cổ chai** — 8 quan hệ không đủ diễn tả tin công nghệ. |
| **Bulk Cypher Ingestion** | M2 | `bulk_insert_nodes()`, `bulk_insert_edges()` | `UNWIND $rows` batch 1000: 1 round-trip thay vì 209. Với Neo4j Aura qua Internet, đây là khác biệt bậc độ lớn dù quy mô còn nhỏ. |
| **Entity Resolution & Union-Find** | M3 | `build_resolution_map()`, `UF` | 55 dòng audit; phát hiện false negative ở **0.8992** — ngưỡng cứng 0.90 cắt ngay giữa vùng xám 0.89–0.92. |
| **Super-node Degree Cap** | M4 | `retrieve_graph_context()` | `supernode_events = 0` trên toàn bộ 12 câu. Cơ chế đã cài, có test, **chưa được dữ liệu thật kiểm chứng**. |
| **LLM-as-a-Judge Evaluation** | M5 | `judge_answer()` | Judge khác nhà cung cấp với generator để tránh self-preference bias. Rationale chỉ đúng nguyên nhân, nhưng judge **tự hỏng 3/25 lần** (`json_validate_failed`) — bản thân judge cũng là một điểm lỗi. |

### 2. Quá trình Debugging & Bài học

**Lỗi phức tạp nhất:** `pandas.Series.name` che mất cột `name` trong `match_seeds()` (cell 3.2).

```python
r = entity_match_store.iloc[k]
r.name      # -> 0          nhãn index của dòng
r["name"]   # -> 'Amazon'   cột name
```

Lỗi này **không sập ở nơi nó xảy ra**: cell 3.2 chạy êm suốt Phần 3 vì graph traversal dùng `seed["id"]` (đúng) chứ không dùng `seed["name"]`. Nó chỉ lộ ra ba cell sau, tại `", ".join()` trong bảng chẩn đoán.

**Cách xử lý thành công:** thứ giúp tìm ra không phải traceback mà là **đối chiếu chéo hai file output** — toàn bộ 10 câu `NameError` trong `graphrag_eval_results.csv` đều có `matched_seeds = 0` trong `golden_coverage.csv`, trong khi các câu `matched_seeds = 0` khác lại chạy bình thường. Từ đó suy ra `match_seeds()` đang **ném exception bị `try/except` nuốt mất**, chứ không phải "không tìm thấy seed".

**Bốn bài học:**
1. Lỗi im lặng đắt hơn lỗi ồn ào — nếu `r.name` ném exception ngay, tôi đã sửa trong 30 giây.
2. Bảng chẩn đoán đáng giá hơn thông báo lỗi.
3. `try/except` quanh lời gọi LLM là con dao hai lưỡi: mỗi khối `except` phải **in ra loại exception**, không được nuốt lặng lẽ.
4. Không dùng attribute access trên pandas Series cho tên trùng API: `name`, `index`, `values`, `size`, `shape`, `dtype`.

**Lỗi lớn thứ hai — lỗi thiết kế thí nghiệm, không phải lỗi code.** Ban đầu golden dataset xây trên **5.000 dòng đầu** của `hackernoon_subset.csv` trong khi pipeline nạp `raw.sample(3000)` từ toàn bộ 62.509 dòng — xác suất một dòng bằng chứng lọt vào corpus chỉ ~5%. Cả hai hệ thống đều không có tài liệu mà đáp án chuẩn trích dẫn, nhưng benchmark vẫn cho ra bảng số trông rất chuyên nghiệp. **Trước khi tin bất kỳ benchmark nào, phải kiểm tra corpus dùng để chấm và corpus dùng để nạp có cùng phạm vi hay không.**

### 3. Kế hoạch Áp dụng vào Đồ án Thực tế

- **Tên đồ án:** Trợ lý tra cứu quy định & văn bản nội bộ doanh nghiệp (policy/compliance QA).
- **Đặc thù & lý do chọn giải pháp:** kho văn bản có nhiều phiên bản chồng lấn theo thời gian và tham chiếu chéo; câu hỏi thường có dạng *"quy định X còn hiệu lực không, bị văn bản nào thay thế?"* — đúng dạng multi-hop theo chuỗi thời gian. **Cần Hybrid, nhưng phải qua cổng kiểm tra định lượng trước:** xây đồ thị thử trên ~10% kho, đo `NO_SEED` trên 30 câu hỏi thật; **nếu `NO_SEED` > 30% thì dừng**, dồn nguồn lực cải thiện Flat RAG. Lợi thế lớn của bài toán này so với lab: quan hệ giữa văn bản (`THAY_THE`, `SUA_DOI`, `THAM_CHIEU`) **có sẵn trong metadata** chứ không phải trích xuất bằng LLM — recall gần 100%, đúng thứ lab này thiếu.
- **Cấu trúc Node & Relation dự kiến:**
  - Nodes: `VanBan(so_hieu, ngay_ban_hanh, ngay_hieu_luc, trang_thai)` · `DieuKhoan(so_dieu, noi_dung)` · `ChuDe(ten)` · `DonVi(ten, cap)`
  - Relations: `THAY_THE`, `SUA_DOI`, `THAM_CHIEU`, `AP_DUNG_CHO` — mọi cạnh bắt buộc có `source_doc_id` **và** `ngay_hieu_luc`, ép ở tầng code bằng `raise`.
- **Chiến lược Entity Resolution:** khớp chính xác theo `so_hieu` đã chuẩn hoá (deterministic, không cần embedding); embedding chỉ dùng cho `ChuDe` và `DonVi`; áp dụng luật OR rút ra từ lab.
- **Chiến lược Super-node:** (a) lọc `trang_thai = 'con_hieu_luc'` **trước khi** áp `LIMIT`; (b) cắt tỉa theo **độ liên quan** thay vì độ mới; (c) ngoại lệ cho câu hỏi lịch sử — lấy **toàn bộ chuỗi `THAY_THE`** thay vì cắt, vì với dạng câu hỏi đó cạnh cũ chính là câu trả lời.

---

## 🎯 TỰ ĐÁNH GIÁ

| Tiêu chí | Điểm tự chấm (1–5) | Ghi chú |
|----------|-------------------|---------|
| Mức độ hiểu bài giảng GraphRAG | 4 | Nắm được cả cơ chế lẫn **điều kiện để nó có lợi**. Điều hiểu rõ nhất không phải "GraphRAG mạnh hơn" mà là "GraphRAG chỉ mạnh hơn khi recall trích xuất đủ cao" — và có số liệu chứng minh. |
| Khả năng kiểm soát AI Coding Agent | 4 | Từ chối 3 đề xuất có lý do rõ ràng, yêu cầu smoke test và bằng chứng in ra. Trừ điểm vì có lúc đã áp dụng bản vá che triệu chứng trước khi tìm ra bug gốc. |
| Chất lượng đồ thị tri thức xây dựng | **2** | Chấm thấp có chủ ý. 1,19 cạnh/node là quá thưa; 50% seed entity vắng mặt; 7/12 câu `NO_SEED`. Provenance sạch tuyệt đối nhưng độ phủ không đạt. Nguyên nhân đã xác định: `EXTRACTION_MAX_CHUNKS = 400` chỉ phủ ~13% corpus. |
| Khả năng phân tích và debug hệ thống | 4 | Truy được lỗi im lặng qua ba cell bằng đối chiếu chéo file output, và xác định đúng nguyên nhân gốc của việc GraphRAG thua (`k=4` vs `k=6`) thay vì dừng ở kết luận hời hợt. |

**Việc cần làm tiếp, theo thứ tự đòn bẩy:**
1. Nâng `EXTRACTION_MAX_CHUNKS` 400 → 1000+ và chạy lại — đòn bẩy lớn nhất, chi phí thấp nhất sau khi đã song song hoá.
2. Sửa `answer_graph_rag()` dùng `k=6` để hybrid là phép cộng chứ không phải phép trừ.
3. Chạy lại 13 câu lỗi sau khi vá bug `Series.name`, để benchmark có đủ n=25 thay vì n=12.
