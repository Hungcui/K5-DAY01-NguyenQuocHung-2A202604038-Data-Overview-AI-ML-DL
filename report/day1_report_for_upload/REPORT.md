# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** GPU (T4) / CPU

**Python / PyTorch / Ultralytics:** Python 3.10 / PyTorch 2.5.1 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=919 ; `class_name`=streetcar ; `rank`=1 ; `score`=0.412458 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
- Record này nói về **cả bức ảnh**. Trong tác vụ Phân loại ảnh (Image Classification), mô hình gán một nhãn phân loại duy nhất cho toàn thể khung hình dựa trên phân phối xác suất tổng thể; đầu ra hoàn toàn không chứa tọa độ bounding box hay vị trí cụ thể của từng đối tượng riêng lẻ trong ảnh.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
- Danh sách tên lớp **đi kèm taxonomy và checkpoint đã huấn luyện sẵn** (ở đây là taxonomy `ImageNet-1K` gồm 1.000 lớp cố định). Mô hình học sâu không thể tự nghĩ ra hay tạo thêm tên lớp mới nằm ngoài taxonomy đã được định nghĩa.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
- Cần ghi đầy đủ cả 3 trường vì:
  + `class_id`: Là mã định danh dạng số giúp lập chỉ mục, truy vấn và lưu trữ hiệu quả trong cơ sở dữ liệu.
  + `class_name`: Giúp con người đọc và hiểu nhãn trực quan.
  + `taxonomy_name`: Xác định ngữ cảnh xuất xứ của dữ liệu (data provenance). Các taxonomy khác nhau (COCO, ImageNet, OpenImages, Pascal VOC) có thể gán cùng một `class_id` cho các lớp khác nhau (ví dụ: ID 0 ở COCO là `person`, còn ở ImageNet là loài cá `tench`). Nếu không có `taxonomy_name`, dữ liệu sẽ bị mơ hồ và không thể tích hợp chính xác vào hệ thống.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
- Guideline cần quy định rõ nguyên tắc xác định **lớp chủ đạo (dominant class)**, bao gồm:
  + Ưu tiên vật thể chiếm tỷ lệ diện tích điểm ảnh lớn nhất trong khung hình.
  + Hoặc ưu tiên đối tượng nằm ở tiền cảnh / trung tâm bức ảnh.
  + Hoặc thiết lập thứ tự phân cấp nghiệp vụ rõ ràng (ví dụ: nếu có phương tiện công cộng trên ray thì ưu tiên gán nhãn phương tiện trước môi trường hạ tầng xung quanh).
6. Vì sao không được lấy `score` của model làm ground truth?
- Vì `score` chỉ là độ tự tin toán học (confidence probability) phản ánh mức độ chắc chắn của mô hình dựa trên trọng số đã học, **không phải nhãn chân lý đúng đắn**. Mô hình vẫn có thể dự đoán sai hoặc bị ảo giác (hallucination) với số điểm `score` rất cao, hoặc nhận diện đúng nhưng `score` thấp. Ground truth phải là sự thật khách quan được con người thẩm định và gán nhãn chuẩn mực theo đúng guideline.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=refrigerator ; `score`=0.876542 ; `bbox_xyxy`=[40.25, 75.12, 220.45, 450.30] ; `bbox_width`=180.20 ; `bbox_height`=375.18
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
- Hộp bao (bounding box) bắt đầu từ điểm góc trên-trái tại tọa độ pixel (40.25, 75.12) và kết thúc tại điểm góc dưới-phải tại tọa độ pixel (220.45, 450.30) tính từ gốc tọa độ (0, 0) ở góc trên cùng bên trái bức ảnh. Hộp có chiều rộng là 180.20 pixel và chiều cao là 375.18 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
- Tại ngưỡng `0.20`: Có 12 prediction (mô hình giữ lại nhiều hộp hơn, bao gồm các vật thể nhỏ hoặc mờ có độ tự tin thấp như cup, bowl, chair, bottle).
- Tại ngưỡng `0.60`: Có 4 prediction (mô hình lọc bỏ hầu hết các dự đoán không chắc chắn, chỉ giữ lại các đối tượng rất rõ ràng như refrigerator, microwave, oven, sink).
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
- Khi hạ ngưỡng, số lượng dự đoán dương tính giả (False Positives - hộp rác, nhận nhầm bóng đổ, hoa văn nền thành vật thể) tăng cao. Người reviewer phải tốn nhiều thời gian và công sức để rà soát, loại bỏ (delete/reject) các hộp sai này và chỉnh sửa các hộp bị gán sai nhãn.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
- *"Hộp bao (bounding box) phải bao khít sát toàn bộ phần nhìn thấy của vật thể (tight bounding box), không để khoảng trống lề vượt quá 2 pixel và tuyệt đối không được cắt lẹm vào bất kỳ bộ phận nhìn thấy nào của đối tượng."*
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
- Trên ảnh `kitchen`, chiếc tủ lạnh hoặc mặt bếp bị cắt mép bởi cạnh khung hình ảnh, và chiếc ghế bị bàn che khuất một phần.
- Em sẽ **viết thành quy tắc trong guideline** cho trường hợp che khuất/cắt mép thông thường: nếu phần nhìn thấy của vật thể $\ge$ 20% và nhận diện chắc chắn được chủng loại, annotator vẽ bounding box bao quanh toàn bộ phần nhìn thấy đó.
- Chỉ đưa lên hỏi Lab Coach (**escalation**) khi vật thể bị che khuất quá nhiều (dưới 10-20%) khiến không thể khẳng định chắc chắn đó là vật thể gì, hoặc ranh giới giữa hai vật bị che lấp hoàn toàn không thể xác định được.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=refrigerator ; `score`=0.884120 ; số điểm polygon=`polygon_point_count`=42 ; 3–5 điểm đầu của `polygon_xy`=[[42.15, 76.50], [45.30, 80.12], [48.60, 110.45], [52.10, 160.20], [55.80, 220.15]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
- So với bounding box (chỉ là khung chữ nhật bao quát thô), polygon cung cấp đường viền hình học chính xác (contour mask) theo biên dạng thực tế của vật thể. Nó phân biệt rõ ràng từng pixel thuộc về đối tượng với pixel thuộc nền hoặc vật lân cận, đặc biệt có giá trị với các vật thể nghiêng, rỗng hoặc có hình dạng bất đối xứng phức tạp.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
- `instance_id` dùng để phân biệt các cá thể (objects) độc lập trong cùng một bức ảnh (ví dụ: cái ghế thứ nhất `kitchen-001` và cái ghế thứ hai `kitchen-002` dù chung class `chair`).
- Nó **không phải class ID** vì không đại diện cho tên loài/chủng loại đối tượng.
- Nó **không phải tracking ID** vì không có tính liên tục theo thời gian để bám vết đối tượng chuyển động xuyên suốt qua các video frame; nó chỉ là mã định danh cục bộ trên một bức ảnh tĩnh trong phạm vi bài lab.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
- *"Đa giác mask phải bám sát chính xác mép ngoài của vật thể; tại các góc cong bo tròn, phải đặt mật độ điểm đủ dày để đường biên mượt mà theo hình dáng thực, không vẽ đường thẳng cắt lẹm vào thân vật thể và không được chấm chờm sang vùng nền."*
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
- Vị trí tiếp giáp giữa mép bồn rửa (sink) và mặt bàn bếp (countertop), hoặc chỗ tiếp xúc giữa hai chai lọ đặt liền nhau.
- Em sẽ **dùng guideline** nếu tài liệu đã có quy tắc phân tầng ưu tiên (vật nào nằm trước/nổi bật hơn sẽ được vẽ mask trọn vẹn phần nhìn thấy).
- Nếu khu vực tiếp xúc bị nhòe nét, bóng đổ tối hoàn toàn không thể phân biệt ranh giới bằng mắt thường, em sẽ chọn **escalation** để Lab Coach thống nhất tiêu chuẩn chung, tránh việc mỗi annotator tự suy đoán một cách cảm tính.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | 1 nhãn phân loại toàn cục cho cả ảnh (`class_id`, `class_name`, `taxonomy_name`) | Ảnh `traffic` có nhiều chủ thể cạnh tranh (tàu điện, ray, xe hơi, người); model dự đoán `streetcar` với độ tin cậy chỉ ~0.41 | Quan sát toàn ảnh, xác định đối tượng chính/nổi bật nhất theo tiêu chí ưu tiên trong guideline | Kiểm tra nhãn có đúng đối tượng chủ đạo theo guideline không; rà soát xem có nhầm lẫn giữa các lớp tương đồng không |
| Phát hiện vật thể | Tập hợp Bounding Box hình chữ nhật `[x_min, y_min, x_max, y_max]` kèm `class_id` | Trên ảnh `kitchen`, nhiều vật nhỏ (chén, chai lọ) đặt san sát dễ bị sót; khi hạ ngưỡng 0.20 xuất hiện nhiều hộp rác nhận nhầm nền | Kéo khung chữ nhật bao khít toàn bộ phần nhìn thấy của từng vật thể, chọn đúng lớp đối tượng | Kiểm tra độ khít (tightness) của khung hộp, đảm bảo không bỏ sót vật thể và loại bỏ các hộp gán nhầm |
| Instance segmentation | Tập hợp các đa giác `polygon_xy` khép kín + `instance_id` + `class_id` + BBox | Đường viền bồn rửa và mặt bếp có độ tương phản thấp; bóng đổ làm nhòe ranh giới giữa các đối tượng chạm nhau | Chấm các điểm đa giác viền khít theo chu vi hình dạng thực của từng cá thể độc lập | Phóng to kiểm tra đường biên polygon có bám sát mép thật không, đảm bảo không bị cắt lẹm, không dính lấn giữa các instance |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
- *"Chỉ sử dụng tập ảnh công khai được chỉ định trong bài lab, tuyệt đối không tải lên hình ảnh cá nhân, thông tin định danh hay bất kỳ dữ liệu nội bộ nào."*
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
- Nếu gặp dữ liệu ngoài phạm vi hoặc nhạy cảm, em sẽ dừng ngay lập tức mọi thao tác và báo cáo cho **Giảng viên / Lab Coach** phụ trách khóa học.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
