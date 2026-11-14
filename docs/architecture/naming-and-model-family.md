# Quy ước tên và model family HAMIC

**Canonical:** có  
**Áp dụng cho:** model repository, config, checkpoint manifest, báo cáo, model card và bài báo

## 1. Cấu trúc tên

Tên hiển thị có dạng:

`HAMIC-<TOPOLOGY_AND_SCALE>[-<VARIANT_OR_STAGE>]`

Tên repository nên viết thường, dùng dấu gạch nối, ví dụ `hamic-e300m`.

| Thành phần | Ý nghĩa | Ví dụ |
|---|---|---|
| `HAMIC` | Prefix duy nhất của family | `HAMIC` |
| `E` | Bidirectional encoder | `E100M`, `E300M` |
| `D` | Causal decoder-only | `D500M`, `D1B` |
| `E…-D…` | Một model encoder–decoder thống nhất | `E100M-D500M` |
| Không hậu tố | Model pretrained/reference ở đúng topology và scale | `HAMIC-D1B` |
| `M` | Modern architecture candidate dùng cho test/ablation; chưa phải model canonical | `HAMIC-D1B-M` |
| `Instruct` | Đã instruction tuning | `HAMIC-D1B-Instruct` |
| `Chat` | Instruct + hội thoại/safety recipe được mô tả | `HAMIC-D1B-Chat` |
| `Embedding` | Encoder đã tuning cho biểu diễn/retrieval | `HAMIC-E300M-Embedding` |

## 2. Tên đã chốt

| Model | Tên canonical | Bản test modern | Loại |
|---|---|---|---|
| Encoder nhỏ | `HAMIC-E100M` | `HAMIC-E100M-M` | Bidirectional encoder khoảng 100M |
| Encoder lớn | `HAMIC-E300M` | `HAMIC-E300M-M` | Bidirectional encoder khoảng 300M |
| Decoder nhỏ | `HAMIC-D500M` | `HAMIC-D500M-M` | Dense causal decoder khoảng 500M; cùng GQA/RoPE/SwiGLU family và tokenizer với D1B |
| Decoder lớn | `HAMIC-D1B` | `HAMIC-D1B-M` | Dense causal decoder khoảng 1B; GQA/RoPE/SwiGLU family |
| Encoder–decoder | `HAMIC-E100M-D500M` | `HAMIC-E100M-D500M-M` nếu có modern ED experiment riêng | Model seq2seq có encoder và decoder liên kết bằng cross-attention/bridge |

`Pretrained/base`, `instruct`, `chat` và `embedding` vẫn là metadata stage/type, nhưng model pretrained không thêm `-Base` vào tên. Bản `-M` chỉ dùng trong experiment registry/checkpoint test; nếu recipe modern thắng và trở thành canonical, artifact phát hành quay về tên không hậu tố với config hash mới.

## 3. Cách hiểu scale

- `100M`, `300M`, `500M`, `1B` là lớp quy mô, không phải cam kết số nguyên chính xác.
- Sai số mục tiêu SHOULD nằm trong khoảng ±5%; ngoại lệ phải giải thích.
- `E100M-D500M` mô tả quy mô backbone encoder và decoder trước khi tính đầy đủ cross-attention, bridge, task head hoặc adapter.
- Do cross-attention thêm tham số, tổng của `E100M-D500M` có thể khoảng 0,65–0,73B tùy topology. Model card MUST ghi `total_parameters`, `trainable_parameters` và cách tính.
- Không dùng `base`/`large` trong tên vì chúng thường gợi cấu hình RoBERTa chuẩn 125M/355M, trong khi HAMIC ghi trực tiếp scale `100M`, `300M`, `500M`, `1B`. Cặp nhỏ/lớn của cùng topology (E100M/E300M, D500M/D1B) đóng vai trò tương tự base/large nhưng tên luôn ghi scale.

## 4. Nhà phát triển và thương hiệu

Metadata SHOULD tách ba trường:

| Trường | Giá trị dự kiến |
|---|---|
| Model family / project | `HAMIC` |
| Developing organization | Chưa dùng tên tổ chức nào; chỉ thêm khi có phê duyệt bằng văn bản |
| Developing team/lab | `HAMIC` (đã chốt, D-001) |

Không được dùng tên và logo tổ chức như một xác nhận chính thức nếu chưa có phê duyệt. `HAMIC` trong tên model là prefix dự án; tư cách pháp nhân, license holder và contact phải ghi riêng.

## 5. Quy tắc revision

- Hiện chưa có artifact nào được train hay phát hành, nên tên model không gắn số phiên bản.
- Identity của mỗi checkpoint/run/release là revision bất biến: Git SHA, model config SHA, tokenizer SHA, corpus manifest SHA và checkpoint SHA.
- Đổi tokenizer, architecture, objective chính hoặc license/release contract tạo model identity mới (run family mới), không nối tiếp lineage cũ.
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
