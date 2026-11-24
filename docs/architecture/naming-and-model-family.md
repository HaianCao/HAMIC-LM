# Quy ước tên và model family HAMIC

> **Quy ước canonical từ 2026-11-17:** encoder dùng size label thay cho số tham
> số trong tên. Các tên `E100M`/`E300M` ở phần lịch sử bên dưới chỉ dùng để
> truy vết checkpoint pilot và không được dùng cho run mới.

| Size | Tên canonical | Profile đầu tiên | Run prefix |
|---|---|---:|---|
| Small | `HAMIC-ES` | 24 lớp × 768, khoảng 151M | `hamic-es-*` |
| Medium | `HAMIC-EM` | 28 lớp × 1.024, khoảng 349M | `hamic-em-*` |
| Large | `HAMIC-EL` | Chưa khóa | `hamic-el-*` |
| Extra Large | `HAMIC-EXL` | Chưa khóa | `hamic-exl-*` |
| Extra Extra Large | `HAMIC-EXXL` | Chưa khóa | `hamic-exxl-*` |

Các mốc quy hoạch khoảng 700M/1,5B/3B chỉ dùng để tính data/compute trong
[kế hoạch mở rộng corpus và scaling](corpus-expansion-dedup-and-scaling.md);
chúng không thay trạng thái “chưa khóa” của topology EL/EXL/EXXL.

`E` nghĩa là bidirectional encoder; `S/M/L/XL/XXL` là size. `M` trong `EM`
nghĩa là **Medium**, không còn là hậu tố “modern candidate”. Kiến trúc thực
(`modernbert`, `roberta`, ...) nằm trong config và manifest.

Model card MUST ghi số tham số chính xác. `HAMIC-ES` và `HAMIC-EM` là model
identity mới vì topology thay đổi; không strict-resume từ checkpoint
`e100m-*`/`e300m-*`. Identifier và Hub path lịch sử giữ nguyên để bảo toàn
provenance. Encoder–decoder mới dùng tên `HAMIC-ES-D500M`; decoder hiện vẫn giữ
`D500M`/`D1B` cho tới khi có quyết định đổi decoder family.

**Canonical:** có  
**Áp dụng cho:** model repository, config, checkpoint manifest, báo cáo, model card và bài báo

## 1. Cấu trúc tên

Tên hiển thị có dạng `HAMIC-<TOPOLOGY_AND_SIZE>[-<STAGE>]`. Tên repository
viết thường và dùng dấu gạch nối, ví dụ `hamic-es`.

| Thành phần | Ý nghĩa | Ví dụ |
|---|---|---|
| `HAMIC` | Prefix duy nhất của family | `HAMIC` |
| `E` | Bidirectional encoder | `ES`, `EM`, `EL` |
| `S/M/L/XL/XXL` | Size label của encoder | `HAMIC-EXL` |
| `D` | Causal decoder-only | `D500M`, `D1B` |
| `E…-D…` | Một model encoder–decoder thống nhất | `ES-D500M` |
| `Instruct`, `Chat`, `Embedding` | Stage/tuning artifact | `HAMIC-EM-Embedding` |

## 2. Tên đã chốt

| Model | Tên canonical | Loại |
|---|---|---|
| Encoder small | `HAMIC-ES` | 24×768, khoảng 151M |
| Encoder medium | `HAMIC-EM` | 28×1.024, khoảng 349M |
| Encoder lớn hơn | `HAMIC-EL`, `HAMIC-EXL`, `HAMIC-EXXL` | Topology chưa khóa |
| Decoder nhỏ | `HAMIC-D500M` | Dense causal decoder khoảng 500M |
| Decoder lớn | `HAMIC-D1B` | Dense causal decoder khoảng 1B |
| Encoder–decoder | `HAMIC-ES-D500M` | Unified seq2seq với cross-attention/bridge đã train |

## 3. Tên lịch sử

`HAMIC-E100M`, `HAMIC-E100M-M`, `HAMIC-E300M`, `HAMIC-E300M-M` và các
run `e100m-*`/`e300m-*` là identifier bất biến của pilot. Báo cáo lịch sử có
thể ánh xạ chúng sang family ES/EM, nhưng không đổi tên artifact và không
strict-resume sang topology canonical mới.

## 4. Nhà phát triển và thương hiệu

Metadata SHOULD tách ba trường:

| Trường | Giá trị dự kiến |
|---|---|
| Model family / project | `HAMIC` |
| Developing organization | Chưa dùng tên tổ chức nào; chỉ thêm khi có phê duyệt bằng văn bản |
| Developing team/lab | `HAMIC` (đã chốt, D-001) |

Không được dùng tên và logo tổ chức như một xác nhận chính thức nếu chưa có phê duyệt. `HAMIC` trong tên model là prefix dự án; tư cách pháp nhân, license holder và contact phải ghi riêng.

## 5. Quy tắc revision

- Các pilot `e100m-*`/`e300m-*` đã tồn tại; giữ nguyên tên và revision của chúng.
- Identity của mỗi checkpoint/run/release là revision bất biến: Git SHA, model config SHA, tokenizer SHA, corpus manifest SHA và checkpoint SHA.
- Đổi tokenizer, architecture, hidden size, số lớp, objective chính hoặc license/release contract tạo model identity mới (run family mới), không nối tiếp lineage cũ.
- Mở rộng recipe/data mà vẫn tương thích tokenizer và architecture tạo revision mới trong cùng model identity.
- Quy ước đánh số phiên bản phát hành chỉ được quyết định khi có checkpoint đầu tiên vượt data, engineering, evaluation và release gate.

## 6. Biến thể encoder–decoder

Chỉ dùng tên `E…-D…` khi đây là **một computation graph thống nhất** và đã huấn luyện phần kết nối. Không dùng tên này cho:

- Một pipeline retrieval dùng encoder độc lập rồi đưa text vào decoder.
- Hai checkpoint đặt cạnh nhau nhưng chưa train cross-attention/bridge.
- RAG system; hệ đó dùng tên sản phẩm/pipeline riêng và ghi hai model thành phần.

Các topology hợp lệ:

1. Cross-attention ở mọi decoder block: chất lượng baseline tốt nhất nhưng thêm nhiều tham số/compute.
2. Cross-attention cách lớp: tiết kiệm tham số, phải ablation.
3. Learned bridge/prefix: nhẹ hơn nhưng không nên mô tả là encoder–decoder chuẩn nếu decoder không cross-attend trực tiếp.

## 7. Metadata tối thiểu của artifact

Mỗi model repository/checkpoint manifest MUST có: tên chuẩn, model family, stage, revision ID, architecture kind, total/trainable parameters, context length, vocab size, tokenizer revision, corpus revision, training tokens, objective, license status, developer/contact, Git SHA và checksum.
