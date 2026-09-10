# Multi-model update — สรุปสิ่งที่เปลี่ยน

อัปเดตเว็บแอปจาก "1 โมเดลตายตัว" ให้รองรับ **หลายโมเดลพร้อมกัน** พร้อม 2 โหมดใช้งาน:

1. **โหมดเลือกโมเดลเดียว** — dropdown เลือกว่าจะให้โมเดลไหนทำนาย (เหมือนของเดิม แต่เลือกได้)
2. **โหมดเปรียบเทียบทุกโมเดล** — อัปโหลดภาพแล้วให้ทุกโมเดลที่โหลดสำเร็จทำนายพร้อมกัน แสดงผลเทียบข้างกัน

## ต้องทำก่อนรัน

วางไฟล์ `.keras` ทั้ง 5 ไฟล์ไว้ที่ `app/models/`:

```
app/models/EfficientNetB0_model.keras
app/models/MobileNetV2_model.keras
app/models/ResNet50V2_model.keras
app/models/Xception_model.keras
app/models/BasicCNN_model.keras
```

ชื่อไฟล์ต้อง**ตรงเป๊ะ**กับที่ประกาศไว้ใน `app/models/registry.py` — ถ้าจะใช้ชื่ออื่น แก้ไฟล์นั้นไฟล์เดียวพอ ไม่ต้องแก้ที่อื่น

มีโมเดลไม่ครบ 5 ตัวก็รันได้ปกติ — ตัวที่หาไฟล์ไม่เจอจะขึ้นสถานะ `error` ที่หน้า `/admin` และจะถูก**ข้ามอัตโนมัติ**ทั้งใน dropdown และโหมดเปรียบเทียบ (ไม่ทำให้ทั้งระบบล่ม)

## ไฟล์ที่เปลี่ยน/เพิ่มใหม่

| ไฟล์ | สถานะ | รายละเอียด |
|---|---|---|
| `app/models/registry.py` | **ใหม่** | จุดเดียวที่ประกาศว่ามีโมเดลอะไรบ้าง (key, ชื่อแสดงผล, ชื่อไฟล์) |
| `app/model_loader.py` | แก้ทั้งไฟล์ | เปลี่ยนจากโหลดโมเดลเดียวเป็น "โหลดทุกโมเดลใน registry ตอน server start" |
| `app/inference.py` | แก้ทั้งไฟล์ | เพิ่ม `predict_image_bytes(..., model_key=...)` และ `predict_image_bytes_compare(...)` |
| `app/routers/predict.py` | แก้ทั้งไฟล์ | เพิ่ม `GET /models`, เพิ่ม field `model` ใน `POST /predict`, เพิ่ม `POST /predict/compare` |
| `app/main.py` | แก้เล็กน้อย | เรียก `warmup_all_models()` แทน `warmup_model()` |
| `app/templates/index.html` | แก้ทั้งไฟล์ | เพิ่มปุ่มสลับโหมด + dropdown เลือกโมเดล + ตารางผลลัพธ์แบบเปรียบเทียบ |
| `app/templates/admin.html` | แก้ทั้งไฟล์ | แสดงสถานะโมเดลทุกตัวเป็นตาราง (ไม่ใช่แค่ตัวเดียว) |
| `app/static/style.css` | เพิ่มบางส่วน | CSS สำหรับปุ่มสลับโหมด, dropdown, ตารางเปรียบเทียบ |
| `app/utils/image_utils.py` | ไม่เปลี่ยน | เหมือนเดิมทุกตัวอักษร |

## เพิ่มโมเดลตัวที่ 6 ในอนาคต ทำยังไง

1. วางไฟล์ `.keras` ใหม่ไว้ใน `app/models/`
2. เพิ่ม 1 entry ใน `app/models/registry.py`:
   ```python
   "densenet121": {
       "display_name": "DenseNet121",
       "file": "DenseNet121_model.keras",
   },
   ```
3. เสร็จ — ไม่ต้องแก้ที่อื่นเลย ระบบจะโหลดอัตโนมัติตอน restart server แล้วโผล่ทั้งใน dropdown และโหมดเปรียบเทียบ

## API ใหม่ที่เพิ่มเข้ามา

- `GET /models` → รายชื่อโมเดลที่โหลดสำเร็จ (key + display_name)
- `POST /predict` → เหมือนเดิม แต่รับ field `model` เพิ่ม (เช่น `model=resnet50v2`) ถ้าไม่ส่งมาจะใช้โมเดลแรกใน registry เป็น default
- `POST /predict/compare` → รับไฟล์ภาพเหมือนกัน แต่คืนผลจาก**ทุกโมเดลที่โหลดสำเร็จ** ต่อ 1 ภาพ

## ข้อควรระวัง

- ทุกโมเดลถูกโหลดเข้า **RAM พร้อมกันทั้งหมด** ตอน server start (ไม่ใช่โหลดตามสั่ง) — ถ้าเครื่อง deploy จริง RAM น้อย (เช่น free tier บาง cloud ที่ให้ ~512MB-1GB) อาจไม่พอ โดยเฉพาะเมื่อมี ResNet50V2 + Xception ที่ตัวใหญ่ ควรเช็ค RAM ที่มีก่อน deploy จริง
- โหมดเปรียบเทียบจะ**ช้ากว่าโหมดเดียวมาก** (ต้องรันทุกโมเดลต่อ 1 ภาพ) เหมาะกับใช้ดูผลทีละภาพ ไม่เหมาะกับอัปโหลดทีละหลายสิบไฟล์พร้อมกัน
