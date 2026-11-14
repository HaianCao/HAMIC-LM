# Đặc tả kiến trúc model HAMIC

**Trạng thái:** reference profile + research-informed candidate; config release chưa khóa, số tham số phải được xác nhận khi khởi tạo runtime  
**Precision mục tiêu:** BF16 training, weight export theo policy phát hành

## 1. Ma trận family

| Model | Vai trò | Baseline objective | Context | Params ước tính |
|---|---|---|---:|---:|
| HAMIC-E100M | Encoder hiệu quả/ablation/deployment | Dynamic MLM | 512; staged 1.024/2.048 | 100,8M (ref) / 102,8M (-M), **measured** |
| HAMIC-E300M | Encoder chất lượng cao | Dynamic MLM | 512; staged 1.024/2.048 | 294,0M (ref) / 291,8M (-M), **measured** |
| HAMIC-D500M | Decoder compute-efficient, scaling bridge tới D1B, decoder của ED | Next-token prediction | 2.048; long-context là stage sau | 486,2M (ref) / 521,8M (-M), **measured** |
| HAMIC-D1B | Sinh văn bản/pretrained model cho SFT | Next-token prediction | 2.048; long-context là stage sau | 1,021B (ref) / 1,011B (-M), **measured** |
| HAMIC-E100M-D500M | Seq2seq/conditional generation | Span corruption + conditional LM | encoder 1.024, decoder 2.048 | ~0,65–0,73B tùy cross-attention |

Decoder có hai scale giống encoder: `D500M` và `D1B`. Lý do: với compute của mốc D1B-5B (khoảng 3e19 FLOPs), model compute-optimal theo Chinchilla vào khoảng 500M; D1B chỉ compute-optimal khi đạt khoảng 20B token. D500M vì vậy là phương án an toàn khi Molab chỉ đủ cho mốc candidate, đồng thời là điểm thứ hai trên scaling curve để quyết định có đầu tư D1B-20B hay không.

### 1.1. Hai profile trong giai đoạn nghiên cứu

- **Canonical/reference (không hậu tố)** giữ cấu hình cũ để có đối chứng tương thích và quy nguyên nhân.
- **Modern candidate (`-M`)** tái đầu tư tham số từ vocab nhỏ hơn vào transformer, đồng thời thử RoPE/GLU/norm hiện đại.
- `-M` là hậu tố cho bản test, không phải stage phát hành lâu dài.
- Nếu `-M` thắng gate tokenizer, quality/compute, stability và downstream, recipe đó trở thành config canonical ở version mới và tên phát hành bỏ `-M`.

Chi tiết bằng chứng và thứ tự ablation nằm trong tài liệu nghiên cứu nội bộ của nhóm (architecture landscape, HAMIC recommendations).

## 2. HAMIC-E100M reference profile

Reference custom ưu tiên head dimension 64 và đủ sâu để làm scaling bridge tới E300M.

| Thuộc tính | Giá trị |
|---|---:|
| Encoder layers | 14 |
| Hidden size | 640 |
| Attention heads | 10 |
| Head dimension | 64 |
| FFN intermediate | 2.560 |
| Vocab | 48.000 gồm special tokens |
| Position | Learned absolute |
| Max position embeddings | 1.024 |
| Activation | GELU |
| Norm | RoBERTa-style LayerNorm |
| Objective | Dynamic MLM, baseline 15% positions |
| Tied embeddings | Có |
| Params | 100.767.360, **measured** (`RobertaForMaskedLM`, gồm bias, LayerNorm và LM head) |

Lý do không gọi là RoBERTa-base: số lớp, hidden và head khác cấu hình RoBERTa-base chuẩn. Compatibility với Transformers có thể dùng RoBERTa-style config, nhưng model card phải công bố config thật.

## 3. HAMIC-E300M reference profile

| Thuộc tính | Giá trị |
|---|---:|
| Encoder layers | 16 |
| Hidden size | 1.152 |
| Attention heads | 18 |
| Head dimension | 64 |
| FFN intermediate | 4.096 |
| Vocab | 48.000 |
| Position | Learned absolute |
| Max position embeddings | 1.024 |
| Activation | GELU |
| Norm | RoBERTa-style LayerNorm |
| Objective | Dynamic MLM |
| Tied embeddings | Có |
| Params | 294.020.992, **measured** |

E300M không mặc định là sentence-embedding model. Biến thể `Embedding` phải được tuning và đánh giá retrieval riêng.

### 3.1. Modern encoder candidates

| Thuộc tính | HAMIC-E100M-M | HAMIC-E300M-M |
|---|---:|---:|
| Layers | 18 | 18 |
| Hidden size | 640 | 1.152 |
| Heads/head dimension | 10 / 64 | 18 / 64 |
| FFN | GeGLU 1.728 | GeGLU 2.752 |
| Vocab target | 20.544 | 20.544 |
| Position | RoPE | RoPE |
| Norm | Pre-LayerNorm | Pre-LayerNorm |
| Attention v1 | Full bidirectional | Full bidirectional |
| Context curriculum | 512 → 1.024 → optional 2.048 | 512 → 1.024 → optional 2.048 |
| Params | 102.813.504, **measured** | 291.806.016, **measured** |

Các candidate này chưa phải config khóa. Vocab 20.544 gồm phần reserved/special và cần đối chứng trực tiếp với BBPE 20.481 của PhoGPT/BamiBERT. Local/global attention không bật ở v1 vì context ngắn; chỉ nghiên cứu khi continuation >2K có dữ liệu/eval thật.

## 4. HAMIC-D1B reference profile

| Thuộc tính | Giá trị |
|---|---:|
| Decoder layers | 22 |
| Hidden size | 2.048 |
| Query heads | 16 |
| Key/value heads | 4 |
| Head dimension | 128 |
| FFN intermediate | 5.120 |
| Vocab | 48.000 |
| Position | RoPE |
| Context v1 | 2.048 |
| Norm | Pre-RMSNorm |
| Activation | SwiGLU |
| Linear bias | Không |
| Input/output embedding | Tied |
| Objective | Causal next-token prediction |
| Params | 1.021.143.040, **measured** (khớp đúng công thức) |

`HAMIC-D1B` là pretrained causal model, không phải chatbot. SFT/instruction tuning tạo artifact khác và không được ghi đè checkpoint pretrained.

### 4.1. Modern decoder candidate

| Thuộc tính | HAMIC-D1B-M |
|---|---:|
| Decoder layers | 24 |
| Hidden size | 2.048 |
| Query/KV heads | 16 / 4 |
| Head dimension | 128 |
| FFN | SwiGLU 4.864 |
| Vocab target | 20.544 |
| Position/context | RoPE / 2.048 |
| Norm | Pre-RMSNorm + QK-Norm |
| Linear bias | Không |
| Embedding | Tied |
| Stability ablation | z-loss off/on |
| Params | 1.011.064.832, **measured** |

Vocab nhỏ hơn tiết kiệm khoảng 56,23M tham số so với 48K. Candidate dùng phần lớn budget đó để thêm hai layer và giảm nhẹ FFN. Phải so `HAMIC-D1B` với `HAMIC-D1B-M` trên proxy parameter/compute-matched; deep–thin có thể tốt hơn về quality nhưng chậm hơn về latency.

## 5. HAMIC-D500M

### 5.1. Reference profile

| Thuộc tính | Giá trị |
|---|---:|
| Decoder layers | 24 |
| Hidden size | 1.280 |
| Query heads | 10 |
| Key/value heads | 2 |
| Head dimension | 128 |
| FFN intermediate | SwiGLU 3.584 |
| Vocab | 48.000 |
| Position/context | RoPE / 2.048 |
| Norm | Pre-RMSNorm |
| Linear bias | Không |
| Input/output embedding | Tied |
| Objective | Causal next-token prediction |
| Params | 486.176.000, **measured** |

Head dimension 128 và GQA giữ giống D1B để dùng chung attention kernel, RoPE config và code path; chỉ đổi width/depth.

### 5.2. Modern candidate `HAMIC-D500M-M`

| Thuộc tính | Giá trị |
|---|---:|
| Decoder layers | 28 |
| Hidden size | 1.280 |
| Query/KV heads | 10 / 2 |
| Head dimension | 128 |
| FFN | SwiGLU 3.584 |
| Vocab target | 20.544 |
| Position/context | RoPE / 2.048 |
| Norm | Pre-RMSNorm + QK-Norm |
| Linear bias | Không |
| Embedding | Tied |
| Stability ablation | z-loss off/on, dùng chung kết luận với D1B nếu proxy nhất quán |
| Params | 521.828.608, **measured** |

Vocab 20.544 tiết kiệm khoảng 35,14M tham số so với 48K; candidate dùng phần đó để thêm bốn layer (deep–thin theo hướng MobileLLM/SmolLM). Ablation phụ: 24×1.536, 12Q/4KV, FFN 3.072 (~522M) nếu proxy cho thấy 2 KV head làm giảm chất lượng.

### 5.3. Compute và vai trò

- Train FLOPs/token ở context 2.048 khoảng 4,0 GFLOP so với khoảng 7,3 GFLOP của D1B-M (**derived**, gồm attention và LM head). Throughput thực tế dự kiến 1,6–1,8× D1B vì width hẹp hơn thường có MFU thấp hơn; phải benchmark.
- Cùng ngân sách session của D1B-5B, D500M đi được khoảng 10B token (~20 token/param).
- D500M dùng chung tokenizer, special-token contract, data pipeline và recipe với D1B; mọi khác biệt recipe phải ghi thành ablation.
- D500M là decoder backbone của `HAMIC-E100M-D500M`.

## 6. HAMIC-E100M-D500M

### 6.1. Baseline đề xuất

- Encoder tái sử dụng topology E100M đã thắng gate (canonical hoặc `-M`).
- Decoder tái sử dụng topology D500M đã thắng gate nhưng bổ sung cross-attention.
- Encoder context tối đa 1.024; decoder context tối đa 2.048 trong v1.
- Objective pretrain chính: span corruption/text reconstruction; conditional next-token loss ở decoder.
- Khởi tạo ưu tiên từ E100M và D500M đã pretrain, sau đó train cross-attention/bridge và tiếp tục joint training. Không mặc định việc ghép hai checkpoint là đủ.
- Encoder hidden 640 và decoder hidden 1.280 khác nhau: cross-attention chiếu K/V trực tiếp từ 640 sang KV heads của decoder; embedding encoder và decoder không tied với nhau dù dùng chung tokenizer.
- Lý do chọn D500M thay vì D1B: tỷ lệ encoder/decoder cân bằng hơn (~1:5 thay vì ~1:10), ED pretrain rẻ hơn gần một nửa trên Molab, và các task text-to-text mục tiêu (tóm tắt, viết lại, sửa lỗi) không cần decoder 1B ở quy mô token hiện tại.

### 6.2. Ba lựa chọn cross-attention

Ước lượng với `E100M-M` (~102,4M không gồm MLM head) và `D500M-M` (~521,8M); mỗi cross-attention block ~3,61M (Q/O 1.280×1.280, K/V 640→2×128, pre-norm, QK-Norm).

| Phương án | Vị trí | Cross-attn params | Tổng params sơ bộ | Đánh đổi |
|---|---|---:|---:|---|
| ED-Full | Mọi decoder block (28) | ~101,0M | ~725M | Baseline rõ ràng, compute cao |
| ED-Alt | Mỗi 2 decoder block (14) | ~50,5M | ~675M | Cân bằng, cần ablation |
| ED-Sparse | Mỗi 4 block (7) hoặc bridge | ~25,2M | ~650M | Nhẹ, rủi ro giảm conditioning |

Con số là ước lượng thiết kế (**derived**), không phải runtime count. Quyết định topology là gate riêng sau khi E100M và D500M ổn định. Biến thể `E100M-D1B` hoặc `E300M-D1B` chỉ mở lại nếu ED-D500M chứng minh lợi ích và còn compute.

### 6.3. Trường hợp sử dụng

Tóm tắt, viết lại có điều kiện, QA sinh, chuyển đổi văn bản và các task text-to-text. Dịch máy chỉ được tuyên bố nếu có corpus song ngữ hợp pháp và benchmark tương ứng.

## 7. Ánh xạ sang kiến trúc Transformers chuẩn (D-014)

Mọi model HAMIC được xuất dưới dạng một class **có sẵn trong thư viện `transformers`**. Người dùng chỉ cần `AutoModelForMaskedLM.from_pretrained(...)`, `AutoModelForCausalLM.from_pretrained(...)` hoặc `pipeline(...)`, giống BERT/RoBERTa, và không cần `trust_remote_code`.

| Profile | Class Transformers | Vì sao khớp |
|---|---|---|
| Encoder reference | `RobertaForMaskedLM` | Post-LN, GELU, vị trí tuyệt đối, LM head kiểu RoBERTa |
| Encoder `-M` | `ModernBertForMaskedLM` | RoPE, GeGLU (`intermediate_size` là độ rộng mỗi nhánh), pre-LN, không bias; `layer_types` toàn `full_attention` để tắt attention cục bộ |
| Decoder reference | `LlamaForCausalLM` | GQA, RoPE, SwiGLU, pre-RMSNorm, không bias, tied embedding |
| Decoder `-M` | `Qwen3ForCausalLM` | Như Llama + QK-Norm trên head dim; tắt sliding window |
| Encoder–decoder | Chưa chọn | Llama/Qwen3 không có cross-attention trong `transformers`; quyết định ở M6 |

Quy ước token đặc biệt dùng chung: `<s>`=0, `<pad>`=1, `</s>`=2, `<unk>`=3, `<mask>`=4, sau đó 32 sentinel `<extra_id_i>` cho span corruption. NFC nằm sẵn trong tokenizer.

Preset nằm ở `configs/models/*.yaml`; số tham số đo bằng `hamic params configs/models/*.yaml` (meta device, tied weight chỉ tính một lần). `rope_theta` trong preset là giá trị tạm, sẽ chốt bằng ablation.

## 8. Bất biến phải test khi triển khai

- Tổng tham số và trainable parameters khớp model manifest.
- Head dimension, tensor shapes, causal mask và padding mask đúng.
- Embedding tying dùng cùng storage khi recipe yêu cầu.
- Encoder không nhìn validation/test; decoder không attend token tương lai.
- Cross-attention mask không đọc padding và giữ đúng document/example boundary.
- Save/load không đổi logits vượt ngưỡng số học đã định.
- Context extension không được bật chỉ bằng thay config; phải có training/evaluation tương ứng.

## 9. Các quyết định còn mở

| Quyết định | Cần bằng chứng |
|---|---|
| Vocab 20.481/20.544/32K/48K | Fertility, embedding cost, downstream, tokenizer/model throughput |
| Encoder R hay M profile | Quality/compute Pareto, stability, downstream, runtime parameter count |
| Encoder max length 512/1.024/2.048 | VRAM, downstream dài, position training và tỷ lệ document đủ dài |
| D1B context 2K/4K | Throughput, data length, long-context eval |
| D1B R hay M profile | Proxy depth/width, QK-Norm, latency và loss stability |
| D500M R hay M profile; 2 hay 4 KV heads | Cùng proxy với D1B; xác nhận ở D500M pilot |
| Có mở D1B-20B không | Scaling curve D-proxy → D500M → D1B pilot, session capacity |
| ED-Full hay ED-Alt (E100M-D500M) | Quality/compute ablation |
| Shared tokenizer cho mọi topology | Fertility và transfer giữa checkpoint |
