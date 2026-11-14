# Báo cáo cá nhân — Data Ingestion Engineering Day 06

## Dữ liệu và pipeline

- **Chủ đề / nguồn:** *Europe and Central Asia Economic Update, Spring 2021: Data, Digitalization, and Governance*, World Bank / IBRD, ISBN 978-1-4648-1698-7.
- **Quyền sử dụng:** CC BY 3.0 IGO; thông tin attribution trong `data/manifest.json`.
- **Corpus:** 30 trang vật lý, trang 19–48; tổng 91.063 ký tự trong snapshot nguồn.
- **Xử lý:** Dùng `pypdf` trích xuất, chuẩn hóa Unicode và whitespace, tạo snapshot JSONL và SHA-256. PDF nguồn được giữ nguyên nội dung.
- **Giới hạn:** Chất lượng văn bản trích xuất phụ thuộc vào cấu trúc PDF; chưa có kết quả audit đầy đủ để định lượng lỗi trích xuất.
- **Embedding:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, Sentence Transformers 5.7.0, CPU, 384 chiều; chi phí ước tính 0 USD.
- **Vector store:** Qdrant Docker server 1.20.0; cosine distance, exact search. Qdrant URL: `http://127.0.0.1:6333`.
- **Metadata:** `document_id`, `source`, `page`, `section` và offset gốc. C sử dụng `section` làm điều kiện lọc khi câu hỏi cung cấp thông tin tương ứng. Nguồn giá trị và quy tắc tạo từng trường được lưu trong pipeline/metadata của corpus.

## Cấu hình và benchmark

| Strategy | Chunking | Khác biệt |
|---|---|---|
| A | Fixed-size 1.200 ký tự, overlap 120 | Baseline, không filter |
| B | Theo đoạn, tối đa 1.200 ký tự | Không filter |
| C | Giống B | Thêm `section` filter khi câu hỏi cung cấp |

- **Benchmark:** 20 câu, gồm 15 dev và 5 test; có 6 câu metadata filter, trong đó 3 câu thuộc test.
- **Evidence:** Đối chiếu kết quả retrieval với gold source/page và evidence quote trong benchmark.
- **Lưu ý test:** q16 ban đầu không đủ evidence để trả lời toàn bộ câu hỏi; q21–q25 được chuẩn bị sau khi xem kết quả sơ bộ. Vì vậy, test có nguy cơ post-hoc bias và không phải holdout độc lập hoàn toàn.
- **Tiêu chí chọn chiến lược:** `run_metadata.json` ghi quy tắc chọn theo Evidence Hit@3 trên dev, tiếp đến grounding rồi tài nguyên. Artifacts không xác nhận thời điểm quy tắc này được chốt; hơn nữa chưa có nhãn grounding để áp dụng đầy đủ quy tắc đó.

## Kết quả

| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count | Ingest (s) | Median search (ms) | Chi phí |
|---|---|---:|---:|---:|---:|---:|---:|
| A | Dev | 5/15 | 9/15 | 96 | 9.660 | 11.710 | $0 |
| A | Test | 1/5 | 2/5 | 96 | 9.660 | 16.652 | $0 |
| B | Dev | 5/15 | 8/15 | 95 | 6.892 | 10.551 | $0 |
| B | Test | 2/5 | 2/5 | 95 | 6.892 | 10.993 | $0 |
| C | Dev | 5/15 | 9/15 | 95 | 0.275* | 10.649 | $0 |
| C | Test | 2/5 | 3/5 | 95 | 0.275* | 11.624 | $0 |

\* C tái sử dụng vectors của B, không embedding lại; 0.275 giây không phải thời gian cold end-to-end tương đương A/B.

| Strategy | Coherence trung bình /2 (n) | Metadata đủ và đúng | Filter tuân thủ | Grounding 0/1/2 |
|---|---|---|---|---|
| A | 1.0/2 (n=10) | 15/15 | N/A | 0/1/2 = 1/2/2; trung bình 1.2/2 (n=5) |
| B | 1.4/2 (n=10) | 15/15 | N/A | 0/1/2 = 0/3/2; trung bình 1.4/2 (n=5) |
| C | 1.4/2 (n=10) | 15/15 | 9/9 mỗi split | 0/1/2 = 0/2/3; trung bình 1.6/2 (n=5) |

- **Audit coherence:** 30/30 candidate (10 vị trí cố định × 3 chiến lược) đã được gán nhãn theo rubric 0–2; điểm trung bình A=1.0, B=1.4, C=1.4. Các lý do theo từng chunk nằm trong `audit_candidates.json`.
- **Audit grounding:** 45/45 hit (5 câu test × top-3 × 3 chiến lược) đã được gán nhãn 0–2 theo mức đủ bằng chứng. Điểm câu hỏi được tổng hợp theo nhãn cao nhất trong top-3; phân bố và trung bình theo chiến lược nằm trong bảng trên. Các nhãn là AI-assisted do Copilot thực hiện theo rubric, chưa được học viên xác nhận và không phải đánh giá độc lập của con người.
- `run.py` tạo và ghi cấu trúc candidate với nhãn ban đầu là `None`; các trường hiện có đủ để lưu nhãn/lý do trong artifact. Nếu chạy lại, script sẽ tạo lại placeholder, nên không chạy lại để bảo toàn audit này. Không cần sửa code.
- **Metadata:** 45/45 hit test (15 hit mỗi chiến lược) khớp document/source/page, page metadata, offsets và nội dung chunk trong corpus snapshot. Gold quote test khớp nguồn 5/5. Filter compliance C là 9/9 trên 3 câu có filter mỗi split; đây là tuân thủ filter, không phải retrieval accuracy.
- A có tổng độ dài chunk 98.983 ký tự do overlap; B/C có 91.063 ký tự. Đây là tổng độ dài chunk, không phải độ dài corpus nguồn.
- Số token: A 24.157; B/C 22.075.

**Run và trace:** `starter/results/final_v3_run_2026_11_14_01/` gồm `metrics.json`, `run_metadata.json`, `resource_metrics.json`, `audit_candidates.json` và `traces.jsonl`.

## Phân tích và kết luận

- **A–B:** A đạt Evidence Hit@3 dev 9/15 so với 8/15 của B; test cả hai cùng 2/5. A dùng overlap và có nhiều chunk hơn, đồng thời thời gian ingest dài hơn.
- **B–C:** C giữ nguyên cách chunking của B và bổ sung filter. C đạt test Evidence Hit@3 3/5 so với 2/5 của B; tuy nhiên test chỉ có 5 câu và có post-hoc bias.
- **Chiến lược:** C cho kết quả test tốt nhất trong run này và thể hiện filter compliance 9/9, nhưng chưa đủ cơ sở kết luận C vượt trội tổng quát. Việc chọn chiến lược còn phụ thuộc yêu cầu về metadata filter và chi phí vận hành.
- **Hai ví dụ retrieval miss (C, test; đối chiếu `traces.jsonl` với gold):** q22 có hit ở trang gold 33 nhưng top-3 không chứa đủ evidence quote; q25 có filter `section=Regional Outlook` tuân thủ trong top-3 nhưng không truy xuất được evidence quote gold ở trang 43. Hai trường hợp này cho thấy filter compliance không đồng nghĩa retrieval hit.
- **Hạn chế:** Test chỉ có 5 câu và có post-hoc bias. Coherence/grounding labels là AI-assisted, chưa được học viên xác nhận. Thời gian ingest C không so sánh trực tiếp được với A/B do tái sử dụng vectors.
- **Cải tiến:** Xây dựng test holdout mới trước khi chạy; hoàn tất chấm coherence/grounding có hướng dẫn và người xác nhận; bổ sung audit lỗi retrieval và đo thời gian cold ingestion thống nhất.

