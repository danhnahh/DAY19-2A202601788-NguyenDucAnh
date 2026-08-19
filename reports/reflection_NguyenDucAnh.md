# Suy Ngẫm Cá Nhân & Kế Hoạch Đồ Án — Lab 19

**Học viên:** Nguyễn Đức Anh · **Mã:** 2A202601788
**Khóa học:** AICB-K34 · Track 3: GraphRAG
**Ngày thực hiện:** 19/08/2026

---

## 1. Mapping bài giảng vào code

| Khái niệm bài giảng | Module | Hàm / khối code | Quan sát thực tế & đánh giá |
|--------------------|--------|-----------------|----------------------------|
| Conservative Coreference | M1 | `resolve_coref_batch()`, `run_coref()` | 400 chunk · 0 batch fail · **152 chunk (38%) bị sửa** · **14 chunk giữ nguyên kèm `unresolved_mentions`**. Con số 14 mới là thứ đáng giá: chunk trung bình chỉ ~45 từ nên đại từ thường có nhiều tiền ngữ ngang nhau; prompt bảo thủ chọn bỏ trống thay vì đoán. |
| Exact Dedup (SHA-1) | M1 | `standardize_news()` — hash `title + text` | Khử trùng lặp chính xác chạy đúng, nhưng **không bắt được near-duplicate**. Chính bộ golden dataset của tôi có câu G5000-36 hỏi về dòng 2532 và 2537 — hai bài Amazon gần như trùng nhau mà SHA-1 coi là khác nhau. Đây là lý do Challenge A (near-dedup) tồn tại. |
| Schema & Allowlist Guard | M2 | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS` | Guard chạy đúng: `extraction_errors_df` rỗng, mọi cạnh đều hợp lệ. Nhưng nó **vừa là lá chắn vừa là nút cổ chai** — 8 quan hệ không đủ diễn tả tin công nghệ, nhiều phát biểu bị loại thẳng (xem `failure_analysis.md` Ca 3). |
| Edge Provenance | M2 | `bulk_insert_edges()` + assert cell 2.4 | `invalid_provenance_edges = 0` trên 124 cạnh. Đây là tiêu chí duy nhất đạt tuyệt đối ngay lần chạy đầu, vì nó được ép ở tầng code (`raise` nếu thiếu cột) chứ không phụ thuộc chất lượng LLM. |
| Bulk Cypher Ingestion | M2 | `bulk_insert_nodes()`, `bulk_insert_edges()`, `batches()` | `UNWIND $rows` theo batch 1000. Ở quy mô 209 node thì lợi ích chưa thấy rõ, nhưng khác biệt kiến trúc là thật: 1 round-trip mạng thay vì 209. Với Neo4j Aura (qua Internet), đây là khác biệt bậc độ lớn. |
| Entity Resolution & Union-Find | M3 | `build_resolution_map()`, `UF`, `merge_guard()` | 55 dòng audit: 3 `MERGE_MANUAL`, 4 `MERGE_VECTOR`, 48 `REJECT_THRESHOLD`, **0 `REJECT_GUARD`**. Phát hiện đắt giá nhất: cặp FIS bị từ chối ở **0.8992** — thiếu đúng 0.0008 so với ngưỡng. Ngưỡng cứng cắt ngay giữa vùng xám. |
| Flat RAG baseline | M4 | `build_flat_index()`, `retrieve_flat_context()` | FAISS `IndexFlatIP` trên 3.000 chunk chuẩn hoá — cosine. Baseline này **mạnh hơn tôi tưởng**: nó thắng GraphRAG ở nhóm multi-hop với 5.0 vs 4.0. |
| Seed Extraction & Fuzzy Match | M4 | `extract_seeds()`, `match_seeds()` | Mắt xích yếu nhất của cả pipeline. **7/12 câu bị `NO_SEED`**, và chỉ **50% (42/84)** seed entity mà golden dataset khai báo tồn tại trong đồ thị. Đây cũng là nơi chứa bug nghiêm trọng nhất (mục 2). |
| BFS + Super-node Degree Cap | M4 | `retrieve_graph_context()`, `recent_edges()` | `supernode_events = 0` trên toàn bộ 12 câu; `collected_edges` cao nhất là 16, cách rất xa `GLOBAL_EDGE_CAP = 250`. **Cơ chế đã cài, đã có test, nhưng chưa được dữ liệu thật kiểm chứng** — đồ thị 1,19 cạnh/node quá thưa. |
| Linearize subgraph | M4 | `textualize()` | Chỗ này cho kết quả bất ngờ theo hướng tích cực. Ở câu G5000-45, chính **định dạng** `A -REL-> B \| date \| chunk` đã dạy mô hình cách trả lời: nó bắt chước lược đồ và đưa ra đáp án đúng về dedup, trong khi Flat RAG bịa ra một node ghép. Giá trị của GraphRAG ở đây là **áp đặt cấu trúc**, không chỉ là cấp dữ kiện. |
| LLM-as-a-Judge | M5 | `judge_answer()`, `comparison_table()` | Judge (Groq `qwen/qwen3.6-27b`) khác nhà cung cấp với generator (OpenAI `gpt-4o-mini`) để tránh self-preference bias. Rationale của judge chất lượng tốt và **chỉ đúng nguyên nhân** ("completely omits NVIDIA due to its absence in the supplied chunks"). Nhưng judge **hỏng 3/25 lần** với lỗi `json_validate_failed` — LLM-as-a-Judge tự nó cũng là một điểm lỗi. |

---

## 2. Quá trình debugging & bài học

**Lỗi kỹ thuật phức tạp nhất gặp phải:** `pandas.Series.name` che mất cột `name` trong `match_seeds()`.

**Cách chẩn đoán:** lỗi này khó ở chỗ nó **không sập ở nơi nó xảy ra**. Chuỗi sự kiện:

1. Cell 3.2 chạy êm suốt Phần 3, không cảnh báo gì.
2. Cell 4.1c sập với `TypeError: sequence item 1: expected str instance, int found`.
3. Sửa vội chỗ sập → 10/25 câu ở cell 4.3 chuyển sang `NameError: name 'r' is not defined`.

Thứ giúp tôi tìm ra là **bảng chẩn đoán `golden_coverage_df`** — bảng này ghi lại `matched_seeds` cho từng câu hỏi *trước khi* chạy evaluation. Đối chiếu ra: toàn bộ 10 câu `NameError` đều có `matched_seeds = 0`, trong khi các câu `matched_seeds = 0` khác lại chạy bình thường. Từ đó suy ra `match_seeds()` **đang ném exception bị `try/except` nuốt mất**, chứ không phải "không tìm thấy seed".

Kiểm chứng cuối cùng bằng ba dòng:

```python
r = pd.DataFrame([{"id":"x1","name":"Amazon","type":"Company"}]).iloc[0]
r.name      # -> 0          nhãn index
r["name"]   # -> 'Amazon'   cột name
```

**Cách xử lý thành công:** đổi sang truy cập bằng khoá `r["name"]`, và sửa `run_evaluation()` để checkpoint **không** resume các dòng có `error` — nếu không, sau khi sửa bug xong, 10 câu lỗi vẫn bị kế thừa nguyên lỗi cũ.

**Bài học rút ra:**

1. **Lỗi im lặng đắt hơn lỗi ồn ào.** Nếu `r.name` ném exception ngay, tôi đã sửa trong 30 giây. Vì nó trả về một số nguyên hợp lệ, lỗi sống sót qua ba cell và làm hỏng 40% bộ kết quả.
2. **Bảng chẩn đoán đáng giá hơn thông báo lỗi.** Không có `golden_coverage_df` thì `NameError` chỉ là một dòng traceback vô nghĩa. Có nó, mẫu hình lộ ra ngay.
3. **`try/except` bao quanh lời gọi LLM là con dao hai lưỡi.** Nó giữ cho pipeline không chết giữa chừng, nhưng cũng biến bug thành "kết quả rỗng". Từ nay, mỗi khối `except` phải **in ra loại exception**, không được nuốt lặng lẽ.
4. **Không dùng attribute access trên pandas Series** cho các tên trùng API của pandas: `name`, `index`, `values`, `size`, `shape`, `dtype`.

**Lỗi lớn thứ hai — không phải lỗi code mà là lỗi thiết kế thí nghiệm.** Ban đầu tôi xây golden dataset trên **5.000 dòng đầu** của `hackernoon_subset.csv`, trong khi pipeline nạp `raw.sample(3000)` từ toàn bộ 62.509 dòng. Xác suất một dòng bằng chứng lọt vào corpus chỉ khoảng 5%. Nghĩa là **cả hai hệ thống đều không có tài liệu mà đáp án chuẩn trích dẫn** — phép so sánh sẽ vô nghĩa mà vẫn cho ra bảng số trông rất chuyên nghiệp. Bài học: trước khi tin bất kỳ benchmark nào, phải kiểm tra **corpus dùng để chấm và corpus dùng để nạp có cùng phạm vi hay không.**

---

## 3. Kiểm soát AI Coding Agent

**Agent giúp được gì nhiều nhất:**
- Chỉ ra sự lệch phạm vi giữa `source_scope` của golden dataset và cách `standardize_news()` lấy mẫu — lỗi này tôi sẽ không tự phát hiện, vì mọi cell đều chạy xanh.
- Chuyển `run_coref()` và `run_extraction()` sang `ThreadPoolExecutor`: **9 phút 05 → 1 phút 54**, giữ nguyên thứ tự chunk và cơ chế cô lập lỗi từng batch.
- Bổ sung `golden_coverage_df` — chính bảng này về sau là thứ giúp tìm ra bug `Series.name`.

**Đề xuất của agent mà tôi từ chối:**

1. **Mở rộng `ALLOWED_RELATIONS`** cho khớp `required_relations` của golden dataset. Sẽ nâng điểm benchmark nhưng vi phạm đặc tả rubric (8 quan hệ cố định). Tôi giữ nguyên và ghi nhận đây là nguồn mất recall trong báo cáo.
2. **Hạ ngưỡng ER xuống 0.88** để cứu cặp FIS. Từ chối vì có tới 48 cặp nằm dưới 0.90 — nới ngưỡng để cứu 1 cặp là mở cửa cho cả một vùng chưa kiểm tra. Giải pháp đúng là luật OR hai chiều.
3. Trong quá trình sửa bug, agent gợi ý **bọc `", ".join()` bằng `str(... or "?")`** cho an toàn. Tôi áp dụng nhưng nhận ra đó chỉ là **che triệu chứng** — nó sẽ biến mọi tên thành `"?"` và giấu luôn bug gốc. Phải sửa `r["name"]` mới là sửa thật.

**Cách tôi verify output của agent thay vì tin ngay:**
- Với mọi thay đổi logic, yêu cầu **smoke test chạy offline** bằng dữ liệu giả trước khi chạy trên Colab (kiểm tra thứ tự chunk sau khi song song hoá, kiểm tra quan hệ ngoài allowlist có bị lọc không).
- Với mọi khẳng định về hành vi pandas, yêu cầu **in ra giá trị thật** (`r.name` → `0`) chứ không chấp nhận lời giải thích.
- Đối chiếu chéo giữa các file output: `golden_coverage.csv` ↔ `graphrag_eval_results.csv` để tìm mâu thuẫn — đây chính là cách bug được phát hiện.

---

## 4. Kế hoạch áp dụng vào đồ án thực tế (Action Plan)

**Tên đồ án:** Trợ lý tra cứu quy định & văn bản nội bộ doanh nghiệp (policy/compliance QA).

**Đặc thù bài toán:** kho văn bản nội bộ có nhiều phiên bản chồng lấn theo thời gian (quy định ban hành → sửa đổi → thay thế), tham chiếu chéo giữa các văn bản, và câu hỏi thường có dạng *"quy định X hiện còn hiệu lực không, bị văn bản nào thay thế?"*.

**Đồ án của tôi có cần GraphRAG không? — Cần, nhưng là Hybrid, và chỉ khi thoả một điều kiện định lượng.**

Bài lab này dạy tôi rằng câu trả lời không phụ thuộc vào việc bài toán "nghe có vẻ multi-hop", mà phụ thuộc vào **tỷ lệ `NO_SEED`**. Ở đây tôi đo được 7/12 câu `NO_SEED` và GraphRAG thua ở nhóm multi-hop — vì đồ thị chỉ có 1,19 cạnh/node. Do đó với đồ án, tôi sẽ đặt **cổng kiểm tra trước khi đầu tư**: xây một đồ thị thử trên ~10% kho văn bản, đo tỷ lệ `NO_SEED` trên 30 câu hỏi thật. **Nếu `NO_SEED` > 30% thì dừng, dồn nguồn lực cải thiện Flat RAG.** Ưu điểm của bài toán này là quan hệ giữa các văn bản (`THAY_THE`, `SUA_DOI`, `THAM_CHIEU`) **có sẵn trong metadata** chứ không phải trích xuất bằng LLM — nên recall gần 100%, đúng thứ mà lab này thiếu.

**Cấu trúc Node & Relation dự kiến:**

| Node label | Thuộc tính chính | Nguồn dữ liệu |
|-----------|-----------------|--------------|
| `VanBan` | `so_hieu`, `ngay_ban_hanh`, `ngay_hieu_luc`, `trang_thai`, `co_quan_ban_hanh` | Metadata hệ thống văn bản (có cấu trúc) |
| `DieuKhoan` | `so_dieu`, `tieu_de`, `noi_dung`, `van_ban_id` | Bóc tách theo cấu trúc điều/khoản |
| `ChuDe` | `ten`, `nhom` | Phân loại thủ công + LLM |
| `DonVi` | `ten`, `cap` | Danh mục tổ chức nội bộ |

| Relation | From → To | Thuộc tính provenance |
|----------|-----------|----------------------|
| `THAY_THE` | `VanBan` → `VanBan` | `ngay_hieu_luc`, `dieu_khoan_can_cu`, `source_doc_id` |
| `SUA_DOI` | `VanBan` → `DieuKhoan` | `ngay_hieu_luc`, `pham_vi_sua`, `source_doc_id` |
| `THAM_CHIEU` | `DieuKhoan` → `DieuKhoan` | `loai_tham_chieu`, `source_doc_id`, `trich_dan` |
| `AP_DUNG_CHO` | `VanBan` → `DonVi` | `ngay_hieu_luc`, `source_doc_id` |

Mọi cạnh bắt buộc có `source_doc_id` **và** `ngay_hieu_luc` — áp dụng đúng bài học `invalid_provenance_edges = 0`, ép ở tầng code bằng `raise` chứ không trông chờ LLM.

**Chiến lược Entity Resolution cho bài toán của tôi:**

Khác hẳn lab này. Văn bản pháp quy có **định danh chuẩn** (`so_hieu` dạng `123/2024/QĐ-XYZ`), nên bước một là **khớp chính xác theo `so_hieu` đã chuẩn hoá** — deterministic, không cần embedding. Embedding chỉ dùng cho `ChuDe` và `DonVi` là nơi tên gọi thật sự biến thiên. Áp dụng **luật OR** đã rút ra từ lab: gộp khi `cosine >= 0.90` **hoặc** (`cosine >= 0.85` **và** `lexical_ratio >= 0.90`), và giữ nguyên `AUDIT_THRESHOLD = 0.60` để luôn có dữ liệu mà điều chỉnh ngưỡng, thay vì đoán.

**Chiến lược xử lý Super-node cho bài toán của tôi:**

Super-node ở đây là chắc chắn xảy ra, khác với lab: một `ChuDe` như "an toàn thông tin" hay một `DonVi` như "Ban Giám đốc" sẽ nối tới hàng nghìn văn bản. Nhưng bài học ở mục Q5 cho thấy **cắt tỉa theo độ mới là sai** với câu hỏi về thứ tự thời gian. Chiến lược của tôi:

1. **Ưu tiên theo hiệu lực trước, không theo ngày ban hành:** lọc `trang_thai = 'con_hieu_luc'` **trước khi** áp `LIMIT`, vì 90% câu hỏi thực tế hỏi về quy định đang áp dụng.
2. **Cắt tỉa theo độ liên quan** cho phần còn lại: xếp hạng cạnh bằng `cosine(embedding(trich_dan), embedding(query))` rồi mới `LIMIT`, thay vì `ORDER BY ngay_ban_hanh DESC`.
3. **Ngoại lệ cho câu hỏi lịch sử:** khi seed extraction nhận diện được ý định về dòng thời gian, chuyển sang lấy **toàn bộ chuỗi `THAY_THE`** thay vì cắt — vì với dạng câu hỏi này, cạnh cũ chính là câu trả lời.

---

## 5. Tự đánh giá

| Tiêu chí | Điểm tự chấm (1–5) | Ghi chú |
|----------|-------------------|---------|
| Mức độ hiểu bài giảng GraphRAG | 4 | Nắm được cả cơ chế lẫn **điều kiện để nó có lợi**. Điều tôi hiểu rõ nhất không phải "GraphRAG mạnh hơn", mà là "GraphRAG chỉ mạnh hơn khi recall trích xuất đủ cao" — và tôi có số liệu để chứng minh. |
| Khả năng kiểm soát AI Coding Agent | 4 | Từ chối 3 đề xuất có lý do rõ ràng, yêu cầu smoke test và bằng chứng in ra thay vì tin lời giải thích. Trừ điểm vì có lúc đã áp dụng bản vá che triệu chứng (`str(... or "?")`) trước khi tìm ra bug gốc. |
| Chất lượng đồ thị tri thức xây dựng | **2** | Đây là điểm yếu nhất và tôi chấm thấp một cách có chủ ý. 209 node / 124 cạnh = 1,19 cạnh/node là quá thưa; 50% seed entity vắng mặt; 7/12 câu `NO_SEED`. Provenance sạch tuyệt đối (0 cạnh lỗi) nhưng độ phủ thì không đạt. Nguyên nhân đã xác định rõ: `EXTRACTION_MAX_CHUNKS = 400` chỉ phủ ~13% corpus. |
| Khả năng phân tích và debug hệ thống | 4 | Truy được một lỗi im lặng qua ba cell bằng cách đối chiếu chéo file output, và xác định đúng nguyên nhân gốc của việc GraphRAG thua (hybrid dùng `k=4` thay vì `k=6`) thay vì dừng ở kết luận hời hợt "GraphRAG kém hơn". |

**Việc cần làm tiếp nếu có thêm thời gian, theo thứ tự đòn bẩy:**
1. Nâng `EXTRACTION_MAX_CHUNKS` 400 → 1000+ và chạy lại — đòn bẩy lớn nhất, chi phí thấp nhất sau khi đã song song hoá.
2. Sửa `answer_graph_rag()` dùng `k=6` để hybrid là phép cộng chứ không phải phép trừ.
3. Chạy lại 13 câu bị lỗi sau khi vá bug `Series.name`, để bảng benchmark có đủ n=25 thay vì n=12.
