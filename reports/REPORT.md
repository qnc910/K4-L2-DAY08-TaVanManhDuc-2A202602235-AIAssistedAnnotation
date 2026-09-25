# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: `Tạ Văn Mạnh Đức`

Công cụ gán nhãn đã dùng: `CVAT`

Mọi con số phải truy được từ `reports/rounds_table.md`, `outputs/selection_round1.csv`, `outputs/metrics_round*.json` hoặc `outputs/round*_diff.md`. Không coi nhãn test do mô hình tạo là chân lý tuyệt đối.

## 1. Dữ liệu và cách chia tập

Tại sao tập chưa gán nhãn (pool) và tập kiểm thử (test set) được chia theo trục thời gian, có vùng
đệm ở giữa, thay vì chia ngẫu nhiên? Nếu chia ngẫu nhiên, số đo trên tập kiểm thử sẽ bị lệch theo
hướng nào, và vì sao?

Trong bài toán giám sát giao thông bằng camera cố định, các phương tiện di chuyển trên đường cao tốc thường xuất hiện trong khung hình qua nhiều giây liên tiếp (tính tương quan thời gian cao - temporal correlation). Nếu chia ngẫu nhiên (random split) các frame vào tập pool/train và tập kiểm thử (test set), các frame liền kề của cùng một chiếc xe sẽ bị chia đôi vào cả hai tập. Khi đó, mô hình chỉ cần "học vẹt" đặc trưng của một chiếc xe cụ thể ở frame train là có thể dễ dàng phát hiện chính chiếc xe đó ở frame test liền kề, dẫn đến hiện tượng rò rỉ dữ liệu (data leakage). Khi bị rò rỉ dữ liệu, các số đo đánh giá như Precision, Recall và AP50 trên tập test sẽ bị lệch theo hướng lạc quan giả tạo (optimistically biased) – điểm số cao bất thường nhưng không phản ánh năng lực tổng quát hóa thực sự của mô hình trên dòng xe mới trong tương lai. Do đó, việc chia tập theo trục thời gian (ví dụ: các frame đầu dùng làm pool để huấn luyện, các frame cuối làm tập test) kết hợp với một vùng đệm thời gian (temporal buffer gap) ở giữa là cực kỳ cần thiết để đảm bảo toàn bộ phương tiện ở tập train đã rời khỏi khung hình trước khi tập test bắt đầu, giữ cho việc đánh giá hoàn toàn độc lập và trung thực.

## 2. Mô hình khởi đầu lạnh (cold start)

Chép dòng vòng 0 từ `rounds_table.md`. Dựa vào `outputs/compare_round0.jpg`, cho biết mô hình khởi
đầu lạnh không khớp nhãn tham chiếu ở những loại xe nào. Độ phủ (recall) theo kích thước xe cho
thấy điều gì? Một trường hợp nào cần người rà lại nhãn tham chiếu trước khi kết luận mô hình sai?

Dòng vòng 0 từ `reports/rounds_table.md`:
| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Dựa vào `outputs/compare_round0.jpg`, mô hình khởi đầu lạnh (yolov8n pretrained trên COCO) không khớp nhãn tham chiếu chủ yếu ở nhóm xe ở xa và xe trong điều kiện ánh sáng yếu hoặc bị che khuất. Độ phủ (Recall tại conf 0.25) phân theo kích thước xe thể hiện sự chênh lệch rõ rệt:
- Xe nhỏ (R small): chỉ đạt 0.182 (18.18%, phát hiện được 12/66 box tham chiếu).
- Xe vừa (R medium): đạt 0.547 (54.73%, 162/296 box).
- Xe lớn (R large): đạt 0.561 (56.10%, 23/41 box).
Số liệu này chứng minh mô hình ban đầu bỏ sót rất nghiêm trọng các xe ở xa (hơn 81% xe nhỏ bị bỏ lọt), vì xe ở xa vào ban đêm chỉ hiển thị như hai đốm sáng mờ ảo hoặc hình khối mờ nhạt, đặc trưng thị giác khác xa dữ liệu COCO ban ngày.
Ngoài ra, nhãn tham chiếu (pseudo-label) của tập test trong bài này được tạo tự động bởi mô hình lớn hơn chứ chưa qua thẩm định thủ công của chuyên gia. Một trường hợp điển hình cần người rà lại nhãn tham chiếu trước khi kết luận mô hình sai là các xe cực xa chỉ có kích thước dưới 16 pixel hoặc các vệt đèn pha phản chiếu trên mặt đường ướt: mô hình tham chiếu có thể đã gán nhầm vệt phản chiếu thành xe, hoặc gán nhãn những chấm sáng không đủ thông tin nhận dạng. Nếu mô hình sinh viên không dự đoán các trường hợp này thì chưa chắc mô hình đã sai, mà có thể chính nhãn tham chiếu bị dương tính giả (false positive).

## 3. Chiến lược chọn mẫu

Giải thích bằng lời công thức `score = W_U·U + W_A·A + W_D·D` và vai trò của `MIN_GAP_S`.
Dẫn ba frame trong `reports/SELECTION.md` và một frame khác để chứng minh cách bạn cân nhắc
độ bất định, ảnh gần trùng và công gán nhãn. Điểm bất định có chứng minh ảnh đó sẽ cải thiện
mô hình không? Vì sao?

Công thức tính điểm ưu tiên chọn frame: `score = W_U·U + W_A·A + W_D·D` (với trọng số mặc định W_U = 0.5, W_A = 0.3, W_D = 0.2) kết hợp 3 tiêu chí:
1. `U` (Độ bất định, chiếm 50% trọng số): Trung bình mức độ bất định của 5 box khó nhất trong ảnh, tính theo công thức entropy nhị phân rút gọn u(c) = 1 - |2c - 1| (đạt giá trị cực đại 1.0 khi độ tin cậy c = 0.5, thể hiện sự phân vân cao nhất của mô hình).
2. `A` (Mức độ mập mờ, chiếm 30% trọng số): Tỷ lệ số lượng box có confidence lấp lửng trong khoảng [0.15, 0.50] so với số lượng box tối đa trong toàn pool. Ảnh càng có nhiều box đang lưỡng lự thì điểm A càng cao.
3. `D` (Độ đa dạng thời gian, chiếm 20% trọng số): Khoảng cách thời gian tới frame đã gán gần nhất, chuẩn hóa chặn ở 10 giây.
Vai trò của `MIN_GAP_S` (2.0 giây): Do góc quay camera là cố định, hai khung hình cách nhau dưới 2 giây ghi lại cùng một đoàn xe với vị trí hầu như không đổi. Quy tắc này loại bỏ các frame gần trùng (near-duplicate) để tránh lãng phí chi phí nhân công gán nhãn mà không giúp AI học thêm được tri thức mới.

Chứng minh sự cân nhắc thông qua các frame đã phân tích trong `reports/SELECTION.md`:
- `frame_0182.jpg` (hạng 1, score 0.9591): Đứng đầu toàn diện với U=0.9182 và A=1.0000 (18 box mập mờ), tập trung nhiều xe tối sát mép trái, xứng đáng được ưu tiên hàng đầu.
- `frame_0099.jpg` (hạng 8, score 0.9063, t=39.6s): Bất định U=0.9460 rất cao, phản ánh khó khăn khi xe bị lùm cây che khuất và xe bị cắt mép dưới phải.
- `frame_0107.jpg` (hạng 14, score 0.8876, t=42.8s): Cách frame_0099 một khoảng thời gian hợp lý (3.2s > MIN_GAP_S), giúp phát hiện và xóa 2 box ảo do vệt đèn phản chiếu trên mặt đường ướt.
- Một frame khác là `frame_0372.jpg` (hạng 6, score 0.9101): Mặc dù có điểm số rất cao, frame này bị loại bỏ vì chỉ cách `frame_0369.jpg` (hạng 2, t=147.6s) đúng 1.2 giây (< 2.0 giây). Việc loại bỏ frame_0372.jpg chứng minh thuật toán ưu tiên tiết kiệm ngân sách rà nhãn, tránh trùng cảnh.

Điểm bất định cao không đồng nghĩa với việc nạp ảnh đó vào train sẽ chắc chắn cải thiện mô hình. Lý do: Độ bất định chỉ phản ánh việc mô hình hiện tại đang thiếu tự tin ở ảnh đó. Nếu frame chứa quá nhiều nhiễu (vệt sáng phức tạp, xe quá xa không thể nhận diện viền chuẩn) hoặc người gán nhãn xử lý không đồng nhất với quy tắc của tập test, mô hình có thể bị loạn đặc trưng hoặc sinh ra overfitting, làm suy giảm hiệu năng tổng quát.

## 4. Các vòng học chủ động (active learning)

Chép bảng từ `rounds_table.md`. Với mỗi vòng, trình bày:

- mức độ bạn đã sửa nhãn gợi ý (số box giữ nguyên, chỉnh sửa, xoá, thêm mới, lấy từ
  `outputs/round*_diff.md`);
- AP50 thay đổi bao nhiêu so với khởi đầu lạnh và so với vòng trước;
- nhóm xe nào tốt lên hoặc xấu đi theo số đo trên cùng tập test.

Dựa vào các ảnh `compare_round*.jpg`, chỉ ra một ca kết quả đổi sau fine-tune (tốt hơn hoặc xấu
đi), cùng lý do có thể kiểm. Dùng `BLIND_SCAN.md`, `REVIEW_LOG.csv` và `round1_diff.md` phân biệt
quan sát độc lập, lỗi pre-label đã sửa và kết quả mô hình sau train. Mô tả một ca khó theo guideline.

Bảng kết quả từ `reports/rounds_table.md`:
| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 341 | 0.555 | -0.217 | 1.000 | 0.199 | 0.331 | 0.000 | 0.203 | 0.488 |

Phân tích Vòng 1:
- Mức độ sửa nhãn gợi ý (lấy từ `outputs/round1_diff.md` / `round1_diff.json`): Trên 12 ảnh được chọn, mô hình ban đầu đề xuất 169 box. Sau khi rà soát và chỉnh sửa trên CVAT, tổng số box chuẩn là 341 box. Trong đó: giữ nguyên (accepted) 111 box (tỷ lệ chấp thuận 65.68%), tinh chỉnh vị trí/kích thước (edited) 39 box, xóa bỏ (deleted - false positive của AI) 19 box, và vẽ thêm mới (added - false negative của AI) 191 box.
- Thay đổi AP50: AP50 vòng 1 đạt 0.555, giảm 0.217 (Δ AP50 = -0.217) so với mức 0.771 của cold start. Điểm nổi bật là độ chính xác (Precision) tăng tuyệt đối lên 1.000 (100% không còn bất kỳ false positive nào tại conf 0.25), nhưng độ phủ (Recall) giảm mạnh từ 0.489 xuống 0.199.
- Diễn biến theo nhóm kích thước xe:
  + Xe nhỏ (R small): giảm từ 0.182 về 0.000 (hoàn toàn không bắt được xe nhỏ trên tập test).
  + Xe vừa (R medium): giảm từ 0.547 xuống 0.203.
  + Xe lớn (R large): giảm từ 0.561 xuống 0.488.
Nguyên nhân là mô hình sau 50 epoch fine-tune trên lô nhỏ 12 ảnh đã trở nên quá thận trọng, học thiên lệch về các xe cự ly gần/rõ nét và đẩy ngưỡng tự tin của các xe ở xa lên rất cao, dẫn đến bỏ sót (FN tăng từ 206 lên 323).

So sánh qua ảnh `outputs/compare_round0.jpg` và `outputs/compare_round1.jpg`:
- Ca kết quả đổi sau fine-tune: Ở `compare_round0.jpg`, mô hình khởi đầu lạnh phát hiện được nhiều xe ở xa với các box có độ tin cậy thấp, đồng thời mắc lỗi vẽ box nhầm vào vệt đèn phản chiếu trên đường. Sang `compare_round1.jpg`, mô hình đã triệt tiêu hoàn toàn các box ảo trên mặt đường (Precision = 1.000), các xe lớn ở cự ly gần được khoanh rất chuẩn xác và ôm khít thân xe. Tuy nhiên, ở các dãy xe xa tít chân trời, mô hình vòng 1 hoàn toàn không vẽ box nào, làm Recall xe nhỏ rơi về 0.000.
- Phân biệt ba nguồn thông tin:
  1. Quan sát độc lập trong `BLIND_SCAN.md`: Tại `frame_0099.jpg`, mắt người đếm được 26 xe và phát hiện 2 vị trí đặc biệt là xe sau lùm cây và xe ở góc dưới bên phải bị cắt mép ảnh.
  2. Lỗi pre-label đã sửa trong `REVIEW_LOG.csv`: Ghi lại cụ thể việc thêm box cho xe bị cắt mép ở `frame_0099.jpg` (added), xóa vệt phản chiếu đèn ở `frame_0107.jpg` (deleted), và kéo ôm khít thân xe tối sát mép trái ở `frame_0182.jpg` (edited).
  3. Kết quả mô hình sau train: Mô hình đã học triệt để việc loại bỏ box ảo (vệt phản chiếu đèn) giống như ta đã xóa trong REVIEW_LOG, nhưng việc nạp thêm nhiều box xe nhỏ/xe khuất trong 12 ảnh train chưa đủ đa dạng để mô hình khái quát hóa cho tập test, khiến mô hình co cụm lại chỉ bắt những xe thật rõ.
- Mô tả một ca khó theo guideline (`GUIDELINE_LABEL.md`): Tình huống xe bị cắt ở mép ảnh (như góc dưới bên phải `frame_0099.jpg`) hoặc xe chỉ nhìn thấy cụm đèn pha và thân xe chìm trong bóng tối. Quy định nêu rõ: chỉ vẽ box bao quanh phần thân xe thực sự nằm trong ảnh hoặc phần thân xe ước lượng được quanh cụm đèn, tuyệt đối không khoanh trùm ra vệt sáng pha chiếu xuống mặt đường.

## 5. Kết luận và giới hạn

Kết quả vòng này so với cold start ra sao? Vì sao bạn dừng hoặc tiếp tục? Đề xuất hai ca còn yếu
hoặc bất định cho vòng sau, kèm chi phí rà nhãn và nguy cơ ảnh gần trùng. Tập kiểm thử chỉ 20 ảnh,
có luật bỏ qua xe quá nhỏ và nhãn tham chiếu do mô hình tạo chưa được rà thủ công; các giới hạn đó
ảnh hưởng thế nào đến kết luận? Nếu AP50 giảm, bạn sẽ kiểm tra điều gì trước khi train thêm?

Kết quả Vòng 1 so với cold start cho thấy sự đánh đổi rõ rệt: Precision đạt mức hoàn hảo 1.000 (loại bỏ hoàn toàn dương tính giả) nhưng AP50 giảm 0.217 (từ 0.771 xuống 0.555) do Recall sụt giảm mạnh. Tôi quyết định tạm dừng (Stop) chu kỳ active learning ở vòng này để đánh giá lại chiến lược thay vì vội vàng gán nhãn tiếp vòng 2. Lý do là việc tiếp tục gán thêm nhãn với cùng cấu hình huấn luyện (50 epoch, full fine-tune trên tập dữ liệu nhỏ) có thể tiếp tục gây hiện tượng co cụm đặc trưng hoặc overfitting, làm lãng phí công sức gán nhãn.

Đề xuất hai ca còn yếu hoặc bất định cho vòng tiếp theo:
1. Xe ở khoảng cách xa (small cars) chỉ hiển thị cụm đèn mờ: Cần bổ sung các frame có đặc trưng xe xa rõ ràng hơn, nhưng chi phí rà nhãn rất cao vì số lượng xe nhỏ nhiều, đòi hỏi phóng to tỉ mỉ từng pixel, đồng thời phải lưu ý quy tắc bỏ qua box dưới 16px.
2. Xe ở dải phân cách bị che khuất một phần bởi cây cối hoặc xe tối màu sát mép: Cần rà kỹ để mô hình học cách phân biệt thân xe với bóng tối.
Cảnh báo: Cần tuân thủ chặt chẽ `MIN_GAP_S >= 2.0s` để tránh chọn phải các ảnh gần trùng, vì camera cố định khiến các frame sát nhau chỉ nhân bản cùng một lỗi mà không giúp ích cho việc huấn luyện.

Ảnh hưởng từ giới hạn của tập kiểm thử:
- Tập test chỉ có 20 ảnh là cỡ mẫu khá nhỏ, khiến các chỉ số thống kê (nhất là trên nhóm xe lớn chỉ có 41 box) rất nhạy cảm với biến động ngẫu nhiên.
- Quy tắc bỏ qua 14 box xe dưới 16 pixel giúp lọc bớt nhiễu, nhưng vẫn còn nhiều xe nhỏ tiệm cận 16 pixel gây khó khăn cho việc phân định ranh giới.
- Quan trọng nhất, nhãn tham chiếu của tập test là do mô hình lớn tạo tự động (pseudo-label) chứ chưa được chuyên gia thẩm định 100%. Do đó, việc AP50 giảm có thể phản ánh sự bất đồng phong cách gán nhãn (label shift) giữa nhãn chuẩn chỉnh nghiêm ngặt theo GUIDELINE_LABEL.md của học viên với nhãn của mô hình tham chiếu (vốn có thể dễ dãi hơn hoặc chứa box rác), chứ không hoàn toàn do mô hình của học viên kém đi.

Nếu AP50 giảm, các điểm tôi sẽ kiểm tra trước khi train thêm:
1. Kiểm tra lại chất lượng nhãn trong `labels/round1/`: Đảm bảo quy chuẩn kích thước box, cách ôm thân xe và việc loại trừ vệt sáng đèn được thực hiện nhất quán trên toàn bộ 12 ảnh.
2. Kiểm tra siêu tham số huấn luyện: Giảm số epoch (hoặc áp dụng early stopping), giảm learning rate, hoặc đóng băng (freeze) các tầng backbone để giữ lại các đặc trưng tổng quát từ COCO, tránh việc mô hình bị quên đặc trưng xe ở xa khi chỉ học trên 12 ảnh.