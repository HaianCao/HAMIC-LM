# Đi qua từng thành phần của mô hình HAMIC

**Trạng thái:** tài liệu giải thích, viết lại từ [đặc tả model](model-specifications.md), và các tài liệu nội bộ về pipeline/tokenizer, khuyến nghị nghiên cứu và toàn cảnh tiếng Việt  
**Đối tượng:** người cần hiểu hoặc trình bày HAMIC mà không phải đọc toàn bộ đặc tả  
**Lưu ý:** khi tài liệu này khác đặc tả, đặc tả thắng. Ví dụ tokenization trong bài là **minh họa**, chưa phải output của tokenizer thật.

---

## 0. Bức tranh tổng thể

Mọi model HAMIC đi qua cùng một dây chuyền. Chỉ phần "khối Transformer" và "mục tiêu huấn luyện" khác nhau giữa encoder, decoder và encoder–decoder.

```
Văn bản tiếng Việt thô
   │  (1) Chuẩn hóa: NFC, giữ dấu, giữ hoa/thường
   ▼
Tokenizer (2): cắt thành token, ~20K từ vựng, không tách từ trước
   │
   ▼
Embedding (3): mỗi token → một vector
   │
   ▼
N × Khối Transformer
   ├─ Vị trí (4): RoPE — model biết token nào đứng trước/sau
   ├─ Attention (5): token nhìn các token khác để hiểu ngữ cảnh
   ├─ FFN (6): xử lý/ghi nhớ thông tin trong từng token
   └─ Norm (7): giữ số liệu ổn định khi train
   │
   ▼
Đầu ra (3): dùng lại bảng embedding để đoán token
   │
   ▼
Mục tiêu huấn luyện (9): đoán chữ bị che (encoder) / đoán chữ tiếp theo (decoder) / khôi phục đoạn bị xóa (ED)
```

Họ model:

| Model | Loại | Dùng để làm gì | Quy mô |
|---|---|---|---:|
| HAMIC-ES | Encoder (đọc hai chiều) | Phân loại, NER, QA trích xuất, chạy nhẹ | ~151M |
| HAMIC-EM | Encoder | Như trên, chất lượng cao hơn | ~349M |
| HAMIC-D500M | Decoder (sinh chữ từ trái sang phải) | Sinh văn bản, nền cho SFT, decoder của ED | ~500M |
| HAMIC-D1B | Decoder | Như trên, lớn hơn | ~1B |
| HAMIC-ES-D500M | Encoder–decoder | Tóm tắt, viết lại, sửa lỗi, QA sinh | Chờ chốt topology cross-attention |

Mỗi model có hai bản: **reference** (cấu hình quen thuộc, dùng làm đối chứng) và **`-M`** (bản hiện đại đang thử). Bản nào thắng trong thí nghiệm thì thành bản chính thức.

### 0.1. Phần nào thật sự dành riêng cho tiếng Việt?

Thành thật mà nói: phần lớn "ruột" Transformer (attention, FFN, norm) là chung cho mọi ngôn ngữ. **Tiếng Việt quyết định chủ yếu ở đầu vào, dữ liệu, mục tiêu huấn luyện và cách đánh giá.** HAMIC cũng dùng đặc điểm tiếng Việt để quyết định cách phân bổ tham số.

| Thành phần | Mức độ riêng cho tiếng Việt | Điểm tiếng Việt chính |
|---|---|---|
| Chuẩn hóa văn bản | **Rất cao** | NFC/NFD, dấu thanh, `hoà`/`hòa`, không dấu |
| Tokenizer | **Rất cao** | Khoảng trắng tách âm tiết chứ không tách từ; vocab ~20K là đủ |
| Embedding | **Cao** (hệ quả của tokenizer) | Vocab nhỏ → dồn tham số vào các lớp |
| Vị trí (RoPE) | Trung bình | Tiếng Việt không chia động từ, nên trật tự từ mang nghĩa |
| Attention | Trung bình | Phải "ghép" các âm tiết thành từ (`máy` + `bay`) |
| FFN, Norm, độ sâu | Thấp (chung) | Thiết kế hiện đại chung |
| Mục tiêu huấn luyện | **Cao** | Che từ khi không có ranh giới từ rõ ràng |
| Dữ liệu và curriculum | **Rất cao** | Nguồn Việt hợp pháp, bộ lọc chỉnh cho tiếng Việt |
| Đánh giá | **Rất cao** | Benchmark Việt, có dấu/không dấu, teencode, NFD |

---

## 1. Chuẩn hóa văn bản

### Là gì?

Trước khi máy "đọc", văn bản được đưa về một dạng thống nhất, để hai chuỗi trông giống nhau thì thực sự giống nhau trong máy.

### Vấn đề riêng của tiếng Việt

**(a) Một chữ có dấu có thể được lưu theo hai cách.** Chữ `ế` có thể là:

- **NFC (dựng sẵn):** 1 ký tự `ế` (U+1EBF).
- **NFD (tổ hợp):** 3 ký tự `e` + dấu mũ + dấu sắc.

Mắt người thấy y hệt nhau, nhưng máy coi là hai chuỗi khác hẳn. Văn bản NFD xuất hiện thật: bộ gõ đặt chế độ "Unicode tổ hợp", tên file trên macOS, văn bản copy từ PDF. Nếu không chuẩn hóa, model sẽ học `Việt` và `Việt` (bản NFD) như hai từ khác nhau.

**(b) Vị trí dấu thanh kiểu cũ và kiểu mới.** `hoà` và `hòa`, `thuỷ` và `thủy` đều được dùng hợp lệ.

**(c) Văn bản không dấu, teencode:** `ko`, `dc`, `j`, `bít`, `toi di hoc`.

### HAMIC áp dụng gì

- **Chuẩn hóa tất cả về NFC.** Bước này bắt buộc.
- **Giữ nguyên:** dấu, hoa/thường, dấu câu, số, công thức, code, chữ tiếng Anh xen kẽ.
- **Không tự ý sửa:** không đổi `hoà`→`hòa`, không bỏ dấu, không viết thường toàn bộ, không sửa chính tả bằng model chưa kiểm toán.
- Văn bản không dấu và teencode được **giữ như thật**, đồng thời có bộ đánh giá riêng để đo model xử lý chúng ra sao.

### Vì sao

Mục tiêu là model hiểu tiếng Việt **như người Việt thực sự viết**. Nếu "làm sạch" quá tay, model sẽ chỉ giỏi văn bản chuẩn và kém hẳn với bình luận mạng xã hội hay văn bản OCR.

---

## 2. Tokenizer: quyết định quan trọng nhất cho tiếng Việt

### Là gì?

Tokenizer cắt văn bản thành các mảnh (token) và gán cho mỗi mảnh một con số. Model chỉ thấy các con số này. Bộ từ vựng (vocab) là danh sách tất cả các mảnh mà model biết.

### Đặc điểm tiếng Việt

Trong tiếng Anh, khoảng trắng gần như tách **từ**. Trong tiếng Việt, khoảng trắng tách **âm tiết**:

```
học sinh   = 1 từ, 2 âm tiết
máy bay    = 1 từ, 2 âm tiết
trường đại học = 1 cụm, 3 âm tiết
```

Có hai cách tiếp cận:

| Cách | Ví dụ đầu vào | Ưu | Nhược |
|---|---|---|---|
| **Tách từ trước** (PhoBERT) | `học_sinh đi học` | Model thấy sẵn "từ" | Cần thêm công cụ tách từ; tách sai thì model học sai; vị trí ký tự cho NER/QA bị lệch; chậm hơn |
| **Đọc thô** (BamiBERT, PhoGPT) | `học sinh đi học` | Đơn giản, không phụ thuộc công cụ ngoài, offset chuẩn | Model phải tự học ghép âm tiết thành từ |

### HAMIC áp dụng gì

**Đường chính: đọc văn bản thô (NFC), dùng byte-level BPE, vocab khoảng 20K.**

| Mã | Cách | Vocab | Vai trò |
|---|---|---:|---|
| T0 | Thô, BBPE giống PhoGPT | 20.481 | Baseline, đã có bằng chứng tốt trên tiếng Việt |
| T1 | Thô, BBPE | **20.544** | Ứng viên chính: căn bội số 64 cho GPU chạy nhanh, dư chỗ cho token đặc biệt |
| T2 | Thô | 32K | Đối chứng: vocab lớn hơn có đáng không? |
| T3 | Thô | 48K | Đối chứng tương thích (không còn là mặc định) |
| T4 | Tách từ trước | 20–32K | Đối chứng: tách từ có giúp task nào không? |

### Vì sao vocab ~20K là đủ cho tiếng Việt

- Tiếng Việt là ngôn ngữ **đơn lập**: từ không biến hình (không có `go/went/gone`, không thêm `-s`, `-ed`). Số âm tiết thực sự dùng trong văn viết chỉ cỡ vài nghìn.
- Vì vậy 20K token đủ để gần như **mỗi âm tiết phổ biến là một token nguyên vẹn**, cộng thêm các cụm hay gặp, tiếng Anh và code.
- Tiếng Anh cần vocab 32K–50K+ vì phải chứa rất nhiều biến thể từ.
- Bằng chứng trực tiếp: PhoGPT-4B (decoder, vocab 20.480) và BamiBERT (encoder, 20.481, đọc thô) đều mạnh trên benchmark Việt.

Minh họa (chưa phải output thật):

```
"Hà Nội là thủ đô của Việt Nam."
→ [Hà] [ Nội] [ là] [ thủ] [ đô] [ của] [ Việt] [ Nam] [.]
   (gần 1 âm tiết = 1 token; dấu cách đi kèm đầu token)
```

### "Byte-level" nghĩa là gì và vì sao quan trọng

Byte-level BPE xây từ **byte** thay vì ký tự. Nhờ vậy **không bao giờ có token "không biết" (UNK)**: emoji 😂, lỗi gõ `Viêt`, teencode `ko bjk`, ký tự lạ từ OCR đều biểu diễn được, chỉ là bị cắt nhỏ hơn. Điều này rất quan trọng với tiếng Việt trên mạng xã hội.

### Tokenizer được chọn thế nào

Không chọn chỉ vì "ít token" hay "ít tham số". Phải qua bộ kiểm tra:

1. Mã hóa rồi giải mã phải ra đúng văn bản gốc.
2. Văn bản NFC và NFD cho cùng kết quả; văn bản không dấu vẫn ổn.
3. Từ nhiều âm tiết, **tên người và địa danh** (`Nguyễn Thị Minh Khai`, `Đắk Lắk`), thuật ngữ không bị cắt vụn.
4. Chữ Anh xen kẽ (`deadline`, `meeting`), URL, emoji, số tiền (`1.500.000đ`), toán, code.
5. Vị trí ký tự (offset) khớp chính xác cho NER/QA.
6. Tốc độ đầu–cuối và hiệu quả thật trên model nhỏ.

Tokenizer được **đóng băng** trước khi train dài. Đổi tokenizer đồng nghĩa với làm model mới.

---

## 3. Embedding và đầu ra

### Là gì?

**Embedding** là một bảng tra: mỗi token có một vector (một dãy số). Kích thước bảng bằng `số token trong vocab × độ rộng model`.

**Đầu ra** làm ngược lại: biến vector thành xác suất cho từng token. HAMIC **dùng chung (tied) một bảng** cho cả đầu vào và đầu ra, nên tiết kiệm được một bảng.

### Liên hệ tiếng Việt: vocab nhỏ thì "tiền" dồn vào bộ não

Vì tiếng Việt chỉ cần ~20K token, bảng embedding nhỏ hơn rất nhiều so với 48K:

| Model | Bảng 48K | Bảng 20.481 | Tiết kiệm |
|---|---:|---:|---:|
| E100M | 30,7M | 13,1M | 17,6M (~17% model) |
| E300M | 55,3M | 23,6M | 31,7M |
| D500M | 61,4M | 26,2M | 35,2M |
| D1B | 98,3M | 41,9M | 56,4M |

Bản `-M` **đem phần tiết kiệm này đổ vào thêm lớp Transformer**, tức là thêm "bộ não" thay vì giữ một bảng tra lớn:

- E100M: 14 → 18 lớp.
- D500M: 24 → 28 lớp.
- D1B: 22 → 24 lớp.

Đây là chỗ đặc điểm ngôn ngữ ảnh hưởng trực tiếp tới kiến trúc.

---

## 4. Mã hóa vị trí: RoPE

### Là gì?

Attention tự nó không biết thứ tự. Nếu không có thông tin vị trí, "chó cắn mèo" và "mèo cắn chó" trông như nhau.

- **Absolute (cũ):** mỗi vị trí 1, 2, 3… có một vector riêng, học trong bảng. Dùng ở bản encoder reference.
- **RoPE (hiện đại):** "xoay" vector theo vị trí, nên model cảm nhận **khoảng cách tương đối** giữa hai token. Dùng ở mọi decoder và encoder `-M`.

### Liên hệ tiếng Việt

Tiếng Việt không chia động từ hay đổi đuôi từ. Ngữ pháp nằm ở **trật tự từ và hư từ**:

- `chó cắn mèo` ≠ `mèo cắn chó`
- `đã` / `đang` / `sẽ` + động từ chỉ thời
- `nhà to` (nhà thì to) ≠ `to nhà` (không tự nhiên)
- Định ngữ đứng sau: `sách của tôi`, `áo màu xanh`

Vì quan hệ "từ này đứng ngay sau/trước từ kia" mang nhiều nghĩa ngữ pháp, mã hóa vị trí tương đối như RoPE rất hợp. RoPE cũng dễ mở rộng ngữ cảnh dài về sau.

---

## 5. Attention: token nhìn nhau

### Là gì?

Mỗi token "hỏi" các token khác: ai liên quan tới tôi? Rồi lấy thông tin từ những token đó. Có nhiều "đầu" (head), mỗi đầu học một kiểu quan hệ.

| Kiểu | Dùng ở | Ý nghĩa |
|---|---|---|
| **Hai chiều** (bidirectional) | Encoder | Token thấy cả trước lẫn sau → hiểu câu tốt nhất |
| **Nhân quả** (causal) | Decoder | Chỉ thấy phía trước → để sinh chữ tiếp theo |
| **Cross-attention** | Decoder của ED | Decoder nhìn sang đầu ra của encoder (văn bản nguồn) |

### Kỹ thuật áp dụng

- **GQA (decoder):** nhiều đầu "hỏi" (Q) dùng chung ít đầu "khóa/giá trị" (K/V): D1B 16Q/4KV, D500M 10Q/2KV. Bộ nhớ khi sinh chữ (KV cache) giảm mạnh mà chất lượng gần như giữ nguyên. Qwen3 và Llama đều dùng.
- **QK-Norm (bản `-M` của decoder):** chuẩn hóa Q và K trước khi nhân, để điểm attention không phình quá lớn làm loss "nổ". Qwen3 và OLMo 2 dùng.
- **Full attention ở 512–2K token:** chưa cần các mẹo attention cục bộ, vì ngữ cảnh chưa dài.

### Liên hệ tiếng Việt

Vì tokenizer đọc thô theo âm tiết, **attention phải tự làm công việc của bộ tách từ**: nhận ra `máy` + `bay` là một từ, `bánh` + `mì` là một món, `Hồ Chí Minh` là một tên. Trên lý thuyết, các lớp đầu sẽ có những head chuyên ghép âm tiết liền kề. Đây là giả thuyết hợp lý, khớp với việc BamiBERT đọc thô vẫn mạnh. HAMIC sẽ kiểm tra bằng đối chứng T4 (tách từ trước).

---

## 6. FFN: "bộ nhớ" trong từng token

### Là gì?

Sau attention, mỗi token đi qua một mạng nhỏ (FFN). Nhiều nghiên cứu cho thấy đây là nơi model "cất" kiến thức: sự kiện, tên riêng, quan hệ.

### Áp dụng

- Bản reference của encoder: **GELU** (kiểu RoBERTa/BERT).
- Bản `-M` của encoder: **GeGLU**. Mọi decoder: **SwiGLU**. Đây là kiểu "có cổng": một nhánh quyết định cho bao nhiêu thông tin đi qua. Chất lượng thường tốt hơn với cùng số tham số.

### Liên hệ tiếng Việt

Phần này **không riêng cho tiếng Việt**. Nhưng kiến thức về Việt Nam (địa danh, lịch sử, pháp luật, văn hóa) chỉ vào được FFN nếu **dữ liệu tiếng Việt chất lượng** đủ nhiều (xem mục 10). Kiến trúc không bù được thiếu dữ liệu.

---

## 7. Chuẩn hóa nội bộ (Norm)

- Encoder reference: LayerNorm kiểu RoBERTa (đặt sau). Encoder `-M`: **Pre-LayerNorm** (đặt trước, ổn định hơn khi model sâu).
- Decoder: **Pre-RMSNorm**, nhẹ và ổn định, chuẩn của Llama/Qwen.
- Bỏ phần lớn bias trong các lớp tuyến tính để đơn giản và nhanh hơn.

Phần này chung, không riêng tiếng Việt. Mục đích là train ổn định trên một GPU mà không bị loss "nổ".

---

## 8. Hình dạng model: sâu hay rộng?

Với cùng số tham số, có thể làm model **rộng** (ít lớp, mỗi lớp to) hoặc **sâu** (nhiều lớp, mỗi lớp nhỏ). Model nhỏ hiện đại (MobileLLM, SmolLM, Qwen3-0.6B) thiên về sâu–mỏng.

| Model | Shape canonical | Ghi chú |
|---|---|---|
| HAMIC-ES | 24 lớp × 768 | ~150,9M; thay family E100M cho run mới |
| HAMIC-EM | 28 lớp × 1.024 | ~348,8M; thay family E300M cho run mới |
| D500M | 28 lớp × 1.280 (`-M`) | Decoder naming chưa đổi |
| D1B | 24 lớp × 2.048 (`-M`) | Decoder naming chưa đổi |

Đánh đổi: sâu hơn thường tốt hơn về chất lượng nhưng sinh chữ chậm hơn một chút. Phải thử nghiệm mới chốt.

---

## 9. Mục tiêu huấn luyện: model học bằng cách nào

### 9.1. Encoder: đoán chữ bị che (MLM)

Che ngẫu nhiên một phần token, bắt model đoán lại:

```
Gốc:   Sinh viên đang ôn thi cuối kỳ .
Che:   Sinh [MASK] đang ôn [MASK] cuối kỳ .
Đoán:  viên, thi
```

- **15%** token bị che là mức chuẩn (đối chứng); **30%** là ứng viên, vì ModernBERT/MosaicBERT cho thấy che nhiều hơn tạo bài tập khó hơn và hiệu quả hơn.
- Che "động": mỗi lần đọc lại câu, vị trí che lại khác.

**Điểm tiếng Việt:** nếu chỉ che `viên` trong `sinh viên`, model dễ đoán vì `sinh` đã gợi ý. Trong tiếng Anh, giải pháp là "che cả từ". Nhưng khi đọc thô, tiếng Việt **không có ranh giới từ chắc chắn**. Vì vậy HAMIC **không mặc định** che cả từ hay che theo cụm; đó là thí nghiệm riêng. Tỷ lệ 30% cũng giúp phần nào, vì nhiều âm tiết của cùng một từ dễ bị che cùng lúc hơn.

Sau đó mới thử **RTD** (kiểu ELECTRA/DeBERTa-v3): một model nhỏ thay token giả, model chính đoán token nào bị thay. Cách này học từ mọi vị trí, nhưng phức tạp hơn.

### 9.2. Decoder: đoán chữ tiếp theo

```
Đầu vào: Thủ đô của Việt Nam là
Đoán:                          Hà → Nội → .
```

Đây là cách mọi LLM được pretrain. Không trộn dữ liệu chat/instruction vào giai đoạn này: bản `-Instruct`/`-Chat` là bước sau và là artifact riêng.

### 9.3. Encoder–decoder: khôi phục đoạn bị xóa (span corruption)

```
Encoder đọc:  Hôm qua <X> đi Đà Lạt <Y> gia đình .
Decoder viết: <X> chúng tôi <Y> cùng
```

Xóa cả **đoạn** (nhiều token liên tiếp) thay vì từng token. **Rất hợp với tiếng Việt**, vì một đoạn tự nhiên bao trọn từ nhiều âm tiết, nên không cần biết ranh giới từ. ViT5 dùng cách này và mạnh ở tóm tắt tiếng Việt.

---

## 10. Độ dài ngữ cảnh

| Model | Ngữ cảnh |
|---|---|
| Encoder | Train 512 trước → tiếp tục 1.024 → (tùy chọn) 2.048 |
| Decoder | 2.048 ở bản đầu; dài hơn là giai đoạn sau |
| ED | Encoder 1.024, decoder 2.048 |

Với vocab ~20K và gần 1 âm tiết ≈ 1 token, 2.048 token tương đương **vài trang văn bản tiếng Việt** (ước lượng; con số chính xác sẽ đo bằng tokenizer thật). HAMIC chỉ công bố độ dài ngữ cảnh **đã thật sự train và đánh giá**, không phải con số ghi trong config.

---

## 11. Dữ liệu và lộ trình học: phần "tiếng Việt" lớn nhất

Kiến trúc chỉ là cái khung; model giỏi tiếng Việt hay không do **dữ liệu** quyết định.

### 11.1. Nguồn

Quy mô, license gate, cross-source dedup, các nguồn HPLT/MADLAD/CulturaX và hàm ý
model scaling được tập trung trong
[kế hoạch mở rộng corpus, dedup và scaling](corpus-expansion-dedup-and-scaling.md).
Tài liệu đó hiện là thiết kế, không phải phê duyệt ingest.

- Dữ liệu nhóm sở hữu hoặc được cấp quyền (ưu tiên cao nhất).
- Wikipedia/Wikisource tiếng Việt.
- FineWeb2 phần tiếng Việt, CulturaX tiếng Việt (sau khi kiểm tra quyền và nội dung).
- Giáo dục, pháp luật, y tế, STEM, code: chỉ nguồn hợp pháp.
- **Không dùng** sách, báo, diễn đàn không rõ quyền, dù có công khai trên mạng.

### 11.2. Lọc dữ liệu: phải chỉnh cho tiếng Việt

Bộ lọc viết cho tiếng Anh sẽ đánh giá sai tiếng Việt. Ví dụ quy tắc "câu toàn từ quá ngắn là rác" sẽ loại nhầm tiếng Việt, vì âm tiết Việt vốn ngắn. HAMIC yêu cầu:

- Kiểm tra bằng người trên cả mẫu **giữ lại** và mẫu **bị loại**.
- Kiểm riêng văn bản có dấu/không dấu, teencode, emoji, chữ Anh xen kẽ, boilerplate kiểu web Việt.
- Lọc trùng lặp (chính xác và gần giống) trên toàn bộ nguồn; lọc thông tin cá nhân (PII) và khóa bí mật.

### 11.3. Tỷ lệ ngôn ngữ

- **VI-100:** 100% tiếng Việt.
- **VI-96:** 96% Việt, 3% Anh, 1% code. Lý do: văn bản Việt thực tế có nhiều chữ Anh xen kẽ (`deadline`, `check`, `file`).

Hai phương án được so bằng thí nghiệm.

### 11.4. Lộ trình ba giai đoạn

1. **G (nền tảng):** dữ liệu tiếng Việt rộng, đã làm sạch.
2. **Q (tinh luyện):** giảm learning rate, tăng dữ liệu chất lượng cao (giáo dục, pháp luật, STEM…), giữ lại một phần dữ liệu cũ để không "quên".
3. **L (ngữ cảnh dài, tùy chọn):** chỉ khi có văn bản dài và bộ đánh giá dài.

Cách này học theo SmolLM2, Qwen3, Sailor2 (Sailor2 giữ khoảng 20% dữ liệu cũ khi tinh luyện cho Đông Nam Á).

### 11.5. Lượng token (thành thật về giới hạn)

| Model | Thử nghiệm | Ứng viên | Mục tiêu nghiên cứu |
|---|---:|---:|---:|
| HAMIC-ES | E100M pilot 7,5B | v0.5: 20B | continuation chỉ mở theo eval gate |
| HAMIC-EM | E300M pilot 3B | v0.5: 20B | continuation chỉ mở theo eval gate |
| D500M | 0,1–1B | 5B | 10B (20B nếu đáng) |
| D1B | 0,1–1B | 5B | 20B |

Để so sánh: PhoGPT-4B dùng ~102B token, TinyLlama ~1T, SmolLM2 ~11T. HAMIC là **pretraining quy mô nghiên cứu**, không tuyên bố ngang các model đó.

---

## 12. Đánh giá: đo bằng thước tiếng Việt

| Model | Đánh giá chính |
|---|---|
| Encoder | ViGLUE, VLUE, bộ 8 task của BamiBERT (NLI, NER y tế COVID, cảm xúc, chủ đề, spam review, ABSA khách sạn/nhà hàng); so với PhoBERT, ViDeBERTa, ViSoBERT, BamiBERT |
| Decoder | Perplexity theo từng miền, VMLU (kiến thức/đọc hiểu), viết tiếp văn bản Việt; so với PhoGPT, Sailor2, Qwen3, Gemma 3, Llama 3.2 cùng cỡ |
| ED | Tóm tắt, viết lại, sửa lỗi, QA sinh; so với BARTpho, ViT5 |

**Bộ kiểm tra độ bền riêng cho tiếng Việt:**

- Có dấu và không dấu (`tôi đi học` và `toi di hoc`).
- NFC và NFD.
- Teencode, emoji, kéo dài chữ (`đẹpppp`).
- Chữ Anh xen kẽ, OCR lỗi, gõ sai dấu.
- Tên người/địa danh Việt.
- **Riêng ED:** kiểm tra model có tự ý trả lời bằng tiếng Anh không ("dịch nhầm", lỗi đã ghi nhận ở mT5).

Model base không bị chấm như chatbot. Bộ test chính thức không được dùng để chọn tokenizer, bộ lọc hay checkpoint.

---

## 13. Tóm tắt từng model

### HAMIC-ES

| Thuộc tính | Giá trị |
|---|---:|
| Lớp × rộng | 24 × 768 |
| Head | 12 × 64 |
| FFN | GeGLU 1.408 |
| Vocab / vị trí | 20.544 / RoPE |
| Norm / mục tiêu | Pre-LN / MLM 30% |
| Params | 150.906.432 |
| Run đầu tiên | cumulative v0.5, 20B token |

### HAMIC-EM

| Thuộc tính | Giá trị |
|---|---:|
| Lớp × rộng | 28 × 1.024 |
| Head | 16 × 64 |
| FFN | GeGLU 2.432 |
| Vocab / vị trí | 20.544 / RoPE |
| Norm / mục tiêu | Pre-LN / MLM 30% |
| Params | 348.796.992 |
| Run đầu tiên | cumulative v0.5, 20B token |

Các profile E100M/E300M cũ là pilot lịch sử và không strict-resume sang ES/EM.

### HAMIC-D500M / D500M-M

| | Reference | `-M` |
|---|---|---|
| Lớp × rộng | 24 × 1.280 | 28 × 1.280 |
| Attention | 10Q / 2KV, head 128 | như bên cạnh + QK-Norm |
| FFN | SwiGLU 3.584 | SwiGLU 3.584 |
| Vocab | 48K | 20.544 |
| Params (đo runtime) | 486,2M | 521,8M |

Chạy **trước** D1B, vì với ngân sách GPU hiện có thì cỡ ~500M là tối ưu về compute. Cũng là decoder của ED.

### HAMIC-D1B / D1B-M

| | Reference | `-M` |
|---|---|---|
| Lớp × rộng | 22 × 2.048 | 24 × 2.048 |
| Attention | 16Q / 4KV, head 128 | như bên cạnh + QK-Norm |
| FFN | SwiGLU 5.120 | SwiGLU 4.864 |
| Vocab | 48K | 20.544 |
| Params (đo runtime) | 1,021B | 1,011B |

Mốc 20B token chỉ mở khi đường scaling D-proxy → D500M → D1B cho thấy đáng đầu tư.

### HAMIC-ES-D500M

- Encoder ES + decoder D500M (bản thắng) + cross-attention.
- Ba phương án: cross-attention ở mọi lớp (~776M), cách một lớp (~724M), thưa (~698M).
- Khởi tạo từ ES và D500M đã train, rồi train tiếp phần nối và toàn bộ model. Ghép hai checkpoint lại **không** tự động thành model hoàn chỉnh.

---

## 14. Những gì **không** áp dụng (và lý do)

| Không dùng | Lý do |
|---|---|
| Tách từ trước làm mặc định | Thêm công cụ phụ thuộc, lệch offset; BamiBERT cho thấy đọc thô vẫn mạnh. Vẫn giữ làm đối chứng |
| Vocab 64K+ | Tốn tham số; tiếng Việt chưa có bằng chứng cần vocab lớn như vậy |
| Bỏ dấu, viết thường | Mất thông tin: `ma`/`má`/`mà`/`mã`/`mạ`/`mả` là các từ khác nhau |
| MoE (mixture of experts) | Phức tạp, chỉ có lợi ở quy mô hạ tầng lớn |
| Ngữ cảnh 32K–128K | Tốn compute; dữ liệu dài tiếng Việt hợp pháp và bộ đánh giá dài chưa đủ |
| Attention cục bộ/toàn cục xen kẽ | Chỉ có ích khi ngữ cảnh > 2K |
| Đổi nhiều thứ cùng lúc | Không biết thay đổi nào gây ra kết quả |

---

## 15. Thuật ngữ nhanh

| Thuật ngữ | Nghĩa ngắn |
|---|---|
| Token | Mảnh văn bản mà model xử lý; với HAMIC thường gần bằng một âm tiết |
| Vocab | Danh sách token model biết |
| NFC/NFD | Hai cách lưu chữ có dấu; HAMIC đưa tất cả về NFC |
| BBPE | Byte-level BPE: thuật toán cắt token từ byte, không có "từ lạ" |
| Embedding | Bảng biến token thành vector |
| Attention | Cơ chế để token lấy thông tin từ token khác |
| GQA | Nhiều đầu hỏi dùng chung ít đầu khóa/giá trị → tiết kiệm bộ nhớ |
| RoPE | Cách mã hóa vị trí bằng phép xoay, cảm nhận khoảng cách tương đối |
| FFN / SwiGLU / GeGLU | Mạng xử lý trong từng token; GLU là dạng có cổng |
| RMSNorm / LayerNorm | Chuẩn hóa giá trị để train ổn định |
| QK-Norm | Chuẩn hóa Q, K để điểm attention không phình to |
| MLM | Che chữ rồi đoán lại (encoder) |
| Span corruption | Xóa đoạn rồi khôi phục (encoder–decoder) |
| Tied embedding | Dùng chung bảng cho đầu vào và đầu ra |
| `-M` | Bản kiến trúc hiện đại đang thử nghiệm |
