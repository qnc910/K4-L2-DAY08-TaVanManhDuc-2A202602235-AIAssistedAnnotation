# Vì sao chọn lô này?

Trong 50 dòng đứng đầu `outputs/selection_round1.csv`, chọn năm frame bạn sẽ ưu tiên nếu chỉ có
ngân sách rà năm ảnh. Ghi tên, điểm, thời điểm, thứ tự và lý do; tối thiểu một quyết định phải xét
ảnh gần trùng hoặc trường hợp model không dự đoán được box: 
Nếu chỉ có ngân sách rà 5 ảnh trong số 50 ứng viên đứng đầu của `outputs/selection_round1.csv`, tôi chọn:
1. `frame_0182.jpg` (hạng 1, điểm 0.9591, thời điểm t=72.8s): Đứng đầu toàn bộ pool với U=0.9182 và A=1.0000 (18 box mập mờ). Ảnh chụp ban đêm rất tối, nhiều xe dồn về mép trái khiến AI rất bất định.
2. `frame_0369.jpg` (hạng 2, điểm 0.9324, thời điểm t=147.6s): Mật độ giao thông rất cao (43 box, 16 box mập mờ), độ bất định của các box khó nhất U=0.9315.
3. `frame_0380.jpg` (hạng 3, điểm 0.9170, thời điểm t=152.0s): Cách frame_0369 là 4.4 giây (đủ giãn cách để không trùng cảnh), có U=0.9340 cao nhất trong nhóm đầu và 40 box xe.
4. `frame_0326.jpg` (hạng 4, điểm 0.9155, thời điểm t=130.4s): Thời điểm khác biệt (t=130.4s), mang tính đại diện cho phân đoạn giữa video với 39 xe và 15 box mập mờ.
5. `frame_0099.jpg` (hạng 8, điểm 0.9063, thời điểm t=39.6s): Ở đây tôi chủ động loại bỏ `frame_0331.jpg` (hạng 5, t=132.4s) vì nó chỉ cách `frame_0326.jpg` (hạng 4, t=130.4s) đúng 2.0 giây – là ngưỡng tối thiểu và cảnh vật gần như trùng lặp nhau. Thay vào đó, tôi chọn `frame_0099.jpg` ở phân đoạn đầu video (t=39.6s) với U=0.9460 rất cao, giúp phân bổ đều dữ liệu trên trục thời gian và tối ưu chi phí gán nhãn cho 5 ảnh.

Ba frame thuộc lô 12 ảnh model chọn và bằng chứng trong CSV/ảnh contact sheet: 
1. `frame_0182.jpg` (hạng 1, score 0.9591, t=72.8s, n_boxes=28, n_ambiguous=18): Điểm cao nhất toàn pool; trên ảnh contact sheet thấy nhiều xe tối sát mép trái có viền không rõ ràng, AI đặt nhãn với độ tin cậy thấp quanh ngưỡng 0.5.
2. `frame_0099.jpg` (hạng 8, score 0.9063, t=39.6s, n_boxes=29, n_ambiguous=14): Bất định U=0.9460 thuộc hàng cao nhất. Bằng chứng trên ảnh cho thấy có xe bị lùm cây che khuất một phần và xe sát mép dưới bên phải chỉ lộ một góc thân xe khiến model bỏ sót hoặc dự đoán thiếu tự tin.
3. `frame_0107.jpg` (hạng 14, score 0.8876, t=42.8s, n_boxes=33, n_ambiguous=15): Được thuật toán chọn do cách frame_0099 một khoảng an toàn (3.2s > 2.0s). Bằng chứng trên contact sheet cho thấy mặt đường ướt có vệt phản chiếu đèn pha gây ra 2 false positive (box ảo) mà model lúng túng.

Một frame có điểm cao nhưng không chọn hoặc một frame có điểm thấp vẫn nên xem, và lý do: 
`frame_0372.jpg` có thứ hạng 6 với điểm số rất cao 0.9101 (U=0.9202, A=0.8333), cao hơn cả các frame đã được chọn như `frame_0312.jpg` (0.9100) hay `frame_0099.jpg` (0.9063). Tuy nhiên, frame này bị thuật toán gạt bỏ (`selected=False`) vì thời điểm t=148.8s chỉ cách `frame_0369.jpg` (hạng 2, t=147.6s) có 1.2 giây, vi phạm quy tắc khoảng cách tối thiểu `MIN_GAP_S = 2.0s`. Do camera quan sát cố định, hai khung hình cách nhau 1.2 giây ghi lại gần như cùng một nhóm xe và bối cảnh đường phố; việc gán nhãn cả hai sẽ gây lãng phí nhân công mà không đem lại tri thức mới cho mô hình.

Điều phép chọn này chưa chứng minh về chất lượng mô hình: 
Điểm số cao trong thuật toán chọn mẫu chỉ phản ánh rằng mô hình hiện tại đang có mức độ bất định lớn (U cao) hoặc sinh ra nhiều bounding box lưỡng lự (A cao). Phép chọn này hoàn toàn chưa chứng minh rằng việc gán nhãn bổ sung các ảnh này sẽ chắc chắn cải thiện chất lượng mô hình (tăng AP50 trên tập test). Nếu ảnh có quá nhiều trường hợp biên gây nhiễu, nhãn gán tay không nhất quán với quy ước test, hoặc số lượng ảnh học quá ít gây hiện tượng overfitting/quên kiến thức cũ, thì mô hình sau fine-tune thậm chí có thể bị giảm độ phủ (recall) hoặc giảm AP50.
