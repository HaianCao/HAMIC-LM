# Kế hoạch mở rộng corpus, dedup và scale model

> **Trạng thái:** tài liệu thiết kế, **chưa phê duyệt để thực thi**. Việc ghi nguồn vào
> tài liệu không cho phép tải, ingest, publish hoặc thay đổi corpus. Mọi nguồn mới phải
> vượt qua gate license, pilot overlap, quality audit và storage/compute trước khi chạy.

Tài liệu này khóa lại các ước tính và nguyên tắc đã thảo luận cho ba câu hỏi:

1. FineWeb-2 hiện tại sẽ cho bao nhiêu token khi hoàn tất đến v0.12?
2. Lượng dữ liệu đó hỗ trợ scale HAMIC-ES/EM/L/XL/XXL tới đâu?
3. Nếu thêm HPLT, MADLAD hoặc CulturaX thì làm sao kiểm soát trùng lặp và quyền sử dụng?

## 1. Baseline FineWeb-2 đến v0.12

Các số đến v0.6 là số đo từ `tokenized_manifest.json`. Các số v0.7–v0.12 là
ước tính, không được trình bày như kết quả đã đo.

| Version | Train token tăng thêm | Train token tích lũy | Trạng thái |
|---|---:|---:|---|
| v0.2 | 4,000B | 4,000B | Đã đo |
| v0.3 | 4,753B | 8,753B | Đã đo |
| v0.4 | 7,436B | 16,189B | Đã đo |
| v0.5 | 7,408B | 23,597B | Đã đo |
| v0.6 | 6,222B | 29,819B | Đã đo |
| v0.7 | ~6,104B | ~35,923B | Ước tính |
| v0.8 | ~6,001B | ~41,924B | Ước tính |
| v0.9 | ~6,309B | ~48,232B | Ước tính |
| v0.10 | ~7,364B | ~55,596B | Ước tính |
| v0.11 | ~7,472B | ~63,067B | Ước tính |
| v0.12 | ~3,646B | **~66,714B** | Ước tính |

Phương pháp ước tính:

- v0.3–v0.6 cho trung bình khoảng **1.089,46 train token trên một dòng nguồn**;
- phần chưa xử lý có khoảng 33.865.644 dòng hiệu dụng;
- v0.12 đã trừ 1.181.000 dòng của `000_00001.parquet` từng được đọc ở v0.2;
- tỷ lệ split giữ nguyên 98% train, 1% validation, 1% test.

Kết quả quy hoạch:

- train: trung tâm **66,7B**, khoảng hợp lý **65,9–67,3B**;
- validation + test: khoảng **1,36B**;
- tổng ba split: trung tâm **68,1B**, khoảng hợp lý **67,3–68,7B**.

Số chính xác chỉ được khóa sau khi v0.12 có manifest cumulative hợp lệ.

## 2. Hàm ý cho model scaling

Heuristic Chinchilla khoảng 20 token/parameter được dùng như một mốc tham khảo cho
compute-optimal dense LM, không phải định luật trực tiếp cho encoder MLM. Với 66,7B
train token, mốc đó tương ứng khoảng 3,34B tham số. Encoder inference-oriented có thể
được train nhiều token/parameter hơn đáng kể; ModernBERT 149M/395M được train trên 2T
token là một tiền lệ rõ ràng.

Nguồn tham khảo:

- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- [ModernBERT](https://arxiv.org/abs/2412.13663)

| Model/profile quy hoạch | Tham số | Token/parameter nếu đọc 66,7B | Mốc 20 token/parameter |
|---|---:|---:|---:|
| HAMIC-ES | 150.906.432 | 442,1 | 3,0B token |
| HAMIC-EM | 348.796.992 | 191,3 | 7,0B token |
| HAMIC-EL candidate | ~700M | 95,3 | 14B token |
| HAMIC-EXL candidate | ~1,5B | 44,5 | 30B token |
| HAMIC-EXXL candidate | ~3B | 22,2 | 60B token |

`EL`/`EXL`/`EXXL` trong bảng chỉ là target quy hoạch, **không khóa topology**.

Với global batch hiện tại là 262.144 input token/optimizer step:

| Ngân sách | Optimizer steps |
|---|---:|
| 20B | 76.294 |
| 30B | 114.441 |
| 40B | 152.588 |
| 60B | 228.882 |
| 66,714B | 254.494 |

Xấp xỉ compute `C ≈ 6ND` khi đọc một lượt 66,7B token:

| Model | Training compute xấp xỉ |
|---|---:|
| HAMIC-ES | 60,4 EFLOP |
| HAMIC-EM | 139,6 EFLOP |
| EL ~700M | 280,2 EFLOP |
| EXL ~1,5B | 600,4 EFLOP |
| EXXL ~3B | 1,20 ZFLOP |

Giới hạn engineering hiện tại:

- trainer chưa có activation checkpointing, FSDP hoặc ZeRO;
- EL là bước scale thực tế tiếp theo trên một GPU 96GB;
- EXL cần benchmark bộ nhớ và gần như chắc chắn cần activation checkpointing;
- EXXL không được coi là runnable trên pipeline hiện tại nếu chưa bổ sung chiến lược
  memory/distributed tương ứng.

## 3. Nguồn mở rộng được xem xét

### 3.1. HPLT 3.0 `vie_Latn` — ưu tiên pilot số 1

Dataset card công bố khoảng 145,4M tài liệu, 142,36B token theo cách đếm của HPLT và
475,6B ký tự. Theo tỷ lệ ký tự/token hiện tại của HAMIC, quy mô trước HAMIC filter và
cross-source dedup có thể vào khoảng 124B HAMIC token.

HPLT cấp phép phần đóng gói dưới CC0 nhưng nói rõ họ không sở hữu nội dung trang web
gốc; nghĩa vụ với nội dung nền vẫn phải được ghi nhận. Xem
[HPLT 3.0 dataset card](https://huggingface.co/datasets/HPLT/HPLT3.0).

Ước tính quy hoạch, chưa được đo: khoảng **40–90B token mới** có thể sống sót sau
quality filter và dedup với FineWeb-2.

### 3.2. MADLAD-400 clean `vi` — ưu tiên sau HPLT

Dataset card công bố khoảng 55M tài liệu clean, 48,7B whitespace token và 228,8B ký
tự. Quy đổi thô theo tokenizer HAMIC cho khoảng 60B token trước cross-source dedup.

Metadata license hiển thị trên Hub và nội dung README cần được đối chiếu, chốt thành
một kết luận license duy nhất trước khi thêm vào source registry. Không ingest chỉ vì
dataset có thể tải công khai. Xem
[MADLAD-400 dataset card](https://huggingface.co/datasets/allenai/MADLAD-400).

Ước tính quy hoạch, chưa được đo: khoảng **10–30B token mới** sau khi HPLT/FineWeb đã
đi trước.

### 3.3. CulturaX `vi` — reservoir overlap cao

CulturaX công bố khoảng 57,6M tài liệu và 55,38B token tiếng Việt. Đây là bản kết hợp,
clean và fuzzy-dedup từ mC4 cùng nhiều release OSCAR. Vì vậy:

- không được tính CulturaX, mC4 và OSCAR như ba nguồn độc lập;
- overlap với FineWeb/HPLT/MADLAD dự kiến cao;
- license kế thừa đồng thời mC4 và OSCAR, không phải một nhãn license đơn giản;
- PII hoặc nội dung nhạy cảm vẫn có thể tồn tại theo cảnh báo của dataset card.

Xem [CulturaX dataset card](https://huggingface.co/datasets/uonlp/CulturaX).

Ước tính quy hoạch, chưa được đo: khoảng **5–20B token mới** nếu CulturaX được chạy
sau FineWeb, HPLT và MADLAD.

### 3.4. Nguồn nhỏ nhưng chất lượng/miền tốt

- Wikipedia mới hơn, Wikisource, Wikibooks, Wikinews: quy mô nhỏ nhưng provenance rõ;
  nội dung Wikimedia thường yêu cầu tuân thủ CC BY-SA/GFDL và attribution tương ứng.
- Dữ liệu luật, giáo dục, y tế, STEM: chỉ thêm khi quyền với văn bản gốc được xác minh,
  không dựa riêng vào tuyên bố của người upload lại.
- Sách, báo, phụ đề, diễn đàn và corpus cộng đồng không rõ quyền: mặc định không dùng.

### 3.5. Tổng quy mô sau mở rộng

Không được cộng thẳng kích thước công bố vì các nguồn web trùng nhau mạnh. Kịch bản
quy hoạch sau cross-source dedup:

| Thành phần | Token train |
|---|---:|
| FineWeb-2 baseline | ~67B |
| HPLT mới | +40–90B |
| MADLAD mới sau HPLT | +10–30B |
| CulturaX mới sau các nguồn trên | +5–20B |
| Tổng khả thi | **~120–205B** |

Khoảng này chỉ dùng để đánh giá khả năng scale. Không dùng làm claim dataset cho tới
khi manifest đo được token thật.

## 4. Dedup contract hiện tại

Mọi delta phải dùng cùng persistent index tại `data/corpus/dedup-index`. Index hiện lưu
cho mỗi tài liệu đã giữ:

- một key 64-bit lấy từ SHA-256 của normalized text cho exact dedup;
- chín key 64-bit cho chín MinHash-LSH bands;
- lịch sử version đã commit.

Cấu hình hiện tại:

- Unicode NFC, HTML unescape, bỏ zero-width và collapse whitespace;
- 128 MinHash permutations;
- 5-syllable shingles trên văn bản lower-case;
- 9 bands × 13 rows, thiết kế quanh threshold 0,8;
- first-seen document wins.

### 4.1. Điều được bảo đảm

- Exact duplicate sau normalize luôn tạo cùng content key và bị loại khỏi corpus mới.
- Duplicate trong cùng batch và duplicate với mọi version đã index đều được kiểm tra.
- Chỉ tài liệu được giữ mới được thêm vào index.
- Index chỉ được save cùng version mới khi build hoàn tất; build lỗi không được coi là
  version đã commit.
- Near-duplicate cluster không bị chia sang nhiều split vì split dùng cluster id.

Key exact 64-bit có xác suất collision rất nhỏ. Collision gây rủi ro loại nhầm tài liệu
khác, không làm một exact duplicate thoát dedup. Nếu quy mô tăng lên hàng trăm triệu
tài liệu, thiết kế sau nên cân nhắc key 128/256-bit để giảm thêm rủi ro này.

### 4.2. Điều không được bảo đảm

MinHash-LSH là xác suất. Với banding hiện tại, xác suất query hit xấp xỉ:

| Jaccard similarity | Xác suất bị bắt |
|---:|---:|
| 70% | 8% |
| 80% | 40% |
| 85% | 69% |
| 90% | 93% |
| 95% | >99% |

Do đó pipeline hiện ưu tiên tránh false positive và bắt bản sao gần nguyên văn, nhưng
chấp nhận có kiểm soát các trường hợp:

- một tài liệu chỉ copy một phần của tài liệu khác;
- nội dung bị đảo thứ tự mạnh;
- paraphrase, dịch hoặc viết lại;
- hai bài về cùng sự kiện nhưng câu chữ khác;
- boilerplate giống nhau trong khi phần chính khác.

Không được mô tả corpus là “zero duplicate”. Claim đúng là: exact-deduplicated sau
normalize và probabilistic near-deduplicated theo cấu hình đã công bố.

### 4.3. First-seen-wins và chất lượng

Incremental index giữ bản xuất hiện trước. Nếu FineWeb có một bản kém sạch và CulturaX
có bản sạch hơn của cùng nội dung, bản CulturaX có thể bị loại. Index hiện chỉ lưu key,
không lưu quality score đủ để thay bản cũ.

Hai lựa chọn phải được quyết định trước corpus v1:

1. **Incremental:** giữ lineage hiện tại; FineWeb thắng mọi overlap. Đơn giản và ổn
   định nhưng không luôn giữ bản chất lượng cao nhất.
2. **Global priority rebuild:** dựng lại corpus/index theo thứ tự nguồn và quality score,
   tốn compute nhưng cho phép nguồn tốt hơn thắng.

Không được âm thầm đổi từ lựa chọn 1 sang 2 vì việc đó thay corpus identity và mọi
manifest downstream.

## 5. “Tải CulturaX mà không cho trùng” nghĩa là gì?

Không thể đồng thời bảo đảm **không truyền byte trùng qua mạng** và **không đọc nội
dung** nếu upstream không cung cấp fingerprint tương thích với index HAMIC.

Có thể bảo đảm ở tầng corpus:

```text
tải/stream một shard
  → normalize + quality/PII filter
  → exact lookup trong persistent index
  → near lookup trong persistent index
  → chỉ ghi tài liệu được giữ
  → token hóa/publish delta đã commit
  → xóa raw shard
  → chuyển sang shard tiếp theo
```

Hệ quả:

- byte trùng vẫn có thể phải tải;
- tài liệu trùng đã phát hiện không tồn tại trong corpus/tokenized cuối;
- không cần giữ toàn bộ CulturaX/HPLT trên ổ cùng lúc;
- source mới phải chạy tuần tự qua một index, không build nhiều version song song;
- URL prefilter có thể tiết kiệm thêm bandwidth nhưng không thay content dedup; cùng nội
  dung có thể tồn tại ở URL khác và index hiện tại không giữ URL của mọi tài liệu cũ.

## 6. Pilot gate trước mọi full ingest

Pilot phải dùng **bản sao** dedup index, không làm bẩn index canonical nếu nguồn bị từ
chối. Quy trình tối thiểu:

1. Pin dataset revision/map/checksum.
2. Hoàn tất license memo và source registry entry ở trạng thái proposal.
3. Chọn mẫu ngẫu nhiên đại diện khoảng 1–2% theo shard/crawl/time/domain.
4. Chạy pipeline với cloned index.
5. Báo riêng: quality rejects, PII redactions, exact duplicates, near duplicates, kept
   documents, kept chars, kept HAMIC tokens và phân bố domain.
6. Audit bằng người cả mẫu giữ và mẫu loại.
7. Chỉ đề xuất full ingest nếu retention và chất lượng bù được bandwidth/storage/compute.

Ngưỡng quy hoạch ban đầu, chưa phải quyết định khóa:

- retention >25%: đáng xem xét full ingest;
- retention 10–25%: chỉ ingest subset chất lượng/domain mới;
- retention <10%: mặc định không tải full.

Trước nguồn overlap cao, nên nghiên cứu thêm chunk/containment dedup để bắt một bài bị
nhúng trong tài liệu dài. Semantic paraphrase vẫn nên được giữ vì thường mang tín hiệu
ngôn ngữ hữu ích, không phải duplicate theo nghĩa storage/training leakage.

## 7. Thứ tự đề xuất và các quyết định còn mở

Thứ tự nghiên cứu, **không phải lệnh thực thi**:

1. Hoàn thành và đo chính xác FineWeb-2 đến v0.12.
2. Freeze manifest và dedup index của corpus v0.x.
3. Quyết định incremental hay global-priority rebuild.
4. Đánh giá/nâng dedup key và containment strategy.
5. HPLT 3.0 pilot bằng cloned index.
6. Chỉ nếu HPLT pass gate mới mở corpus v1.x theo batch 8–12GB.
7. MADLAD pilot sau khi license được chốt.
8. CulturaX pilot cuối; không cộng lại mC4/OSCAR như nguồn riêng.

Các câu hỏi chưa khóa:

- topology chính xác của EL/EXL/EXXL;
- full-run token budget của ES/EM có chuyển từ 20B lên một lượt corpus hay không;
- incremental first-seen-wins hay global quality-priority rebuild;
- threshold/recall mục tiêu của near và containment dedup;
- license memo cuối cho MADLAD/CulturaX;
- retention thực tế của từng nguồn mới.

