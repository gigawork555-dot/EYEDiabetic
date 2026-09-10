# คู่มือรันแอปด้วย Docker + แชร์ให้เครื่องอื่นใน LAN ใช้งาน

ใช้กับเครื่อง COM-GARAGE (CPU-only, ไม่มี GPU) แชร์ให้เครื่องอื่นในวงแลน/wifi เดียวกันใช้งานได้

---

## ⚠️ ก่อนเริ่ม: Docker ไม่ได้ทำให้ "ประมวลผลเร็วขึ้น"

Docker เป็นแค่กล่องแยก environment ให้พกพา/ติดตั้งง่าย ไม่ได้เร่งความเร็วการคำนวณ — บน CPU-only
ความเร็วจะ**ใกล้เคียงหรือช้ากว่ารันตรงๆ ด้วย `python run.py` เล็กน้อย** (มี overhead ของ container
นิดหน่อย) ถ้าอยากได้ความเร็วเพิ่มจริงโดยไม่มี GPU ดูหัวข้อ **"ทำให้เร็วขึ้นโดยไม่ใช้ GPU"** ท้ายไฟล์นี้

ประโยชน์ที่ได้จริงจาก Docker ในเคสนี้คือ **พกพาง่าย + แชร์ให้เครื่องอื่นใช้ได้สะดวก** ไม่ใช่ความเร็ว

---

## ขั้นตอนที่ 1 — ติดตั้ง Docker Desktop (ถ้ายังไม่มี)

ดาวน์โหลดจาก https://www.docker.com/products/docker-desktop/ ติดตั้งแล้วเปิดโปรแกรมทิ้งไว้
(Docker Desktop บน Windows ใช้ WSL2 backend โดยปริยาย ซึ่งรองรับการ map พอร์ตออกสู่ LAN ได้ปกติ
ไม่ต้องตั้งค่าอะไรเพิ่มสำหรับเรื่องนี้)

## ขั้นตอนที่ 2 — จัดโครงสร้างโฟลเดอร์ให้พร้อม

ให้แน่ใจว่ามีไฟล์ `.keras` ทั้ง 5 ไฟล์อยู่ที่ `app/models/` แล้ว (ตามที่ทำไว้ก่อนหน้า) โครงสร้างควรเป็น:

```
โปรเจกต์/
├── docker-compose.yml      <- ไฟล์ใหม่ที่ให้มา
├── app/
│   ├── Dockerfile
│   ├── .dockerignore       <- ไฟล์ใหม่ที่ให้มา
│   ├── main.py
│   ├── models/
│   │   ├── registry.py
│   │   ├── EfficientNetB0_model.keras
│   │   ├── MobileNetV2_model.keras
│   │   ├── ResNet50V2_model.keras
│   │   ├── Xception_model.keras
│   │   └── BasicCNN_model.keras
│   └── ...
```

## ขั้นตอนที่ 3 — Build + รัน

เปิด PowerShell หรือ Command Prompt ที่โฟลเดอร์โปรเจกต์ (ที่มี `docker-compose.yml`) แล้วรัน:

```powershell
docker compose up --build -d
```

- `--build` = สร้าง image ใหม่จาก Dockerfile (ครั้งแรกจำเป็น ครั้งต่อไปถ้าไม่ได้แก้โค้ดข้ามได้)
- `-d` = รันเบื้องหลัง (detached) ไม่ค้าง terminal ไว้

เช็คว่าโมเดลโหลดครบไหม (ดู log):

```powershell
docker compose logs -f
```

ควรเห็นบรรทัดแบบ `Model 'efficientnetb0' loaded OK (version=... updated=...)` ครบทั้ง 5 ตัว
กด `Ctrl+C` เพื่อออกจากโหมดดู log (container ยังรันอยู่เบื้องหลังต่อ ไม่ได้หยุด)

## ขั้นตอนที่ 4 — ทดสอบบนเครื่องตัวเอง

เปิดเบราว์เซอร์ไปที่ `http://localhost:8000` ควรเห็นหน้าแอปตามปกติ

## ขั้นตอนที่ 5 — หา IP ของเครื่อง COM-GARAGE ในวง LAN

เปิด PowerShell แล้วรัน:

```powershell
ipconfig
```

หาส่วน **"Wireless LAN adapter Wi-Fi"** (หรือ "Ethernet adapter" ถ้าต่อสายแลน) แล้วดูบรรทัด
**IPv4 Address** เช่น `192.168.1.xxx` — นี่คือ IP ที่เครื่องอื่นในวงเดียวกันจะใช้เข้าถึง

## ขั้นตอนที่ 6 — เปิด Windows Firewall ให้พอร์ต 8000

โดย default **Windows Firewall จะบล็อกการเชื่อมต่อเข้ามาจากเครื่องอื่น** ต้องเปิดพอร์ตก่อน
รัน PowerShell **แบบ Administrator** (คลิกขวา → Run as administrator) แล้วรันคำสั่งนี้ครั้งเดียว:

```powershell
netsh advfirewall firewall add rule name="DR Screening App" dir=in action=allow protocol=TCP localport=8000
```

หรือทำผ่าน GUI: `Control Panel → Windows Defender Firewall → Advanced settings → Inbound Rules
→ New Rule → Port → TCP → Specific local ports: 8000 → Allow the connection`

## ขั้นตอนที่ 7 — เข้าใช้งานจากเครื่องอื่นในวง LAN

จากเครื่องอื่น (คอม/มือถือที่ต่อ wifi เดียวกัน) เปิดเบราว์เซอร์ไปที่:

```
http://<IP ที่หาได้จากขั้นตอนที่ 5>:8000
```

เช่น `http://192.168.1.106:8000`

---

## คำสั่งที่ใช้บ่อย

| ต้องการ | คำสั่ง |
|---|---|
| หยุด server | `docker compose down` |
| รันใหม่ (ไม่ build ใหม่) | `docker compose up -d` |
| build ใหม่หลังแก้โค้ด | `docker compose up --build -d` |
| ดู log สด | `docker compose logs -f` |
| เช็คว่า container รันอยู่ไหม | `docker compose ps` |
| รีสตาร์ท | `docker compose restart` |

---

## ทำให้เร็วขึ้นโดยไม่ใช้ GPU

ถ้าอยากให้ตอบสนองไวขึ้นจริง ลองทำสิ่งเหล่านี้แทน (เรียงจากง่ายสุดไปยาก):

### 1. โหมดเปรียบเทียบช้าเพราะรัน 5 โมเดลต่อภาพ — ถ้าไม่จำเป็นต้องมีครบ 5 ตัวเสมอ
แก้ `app/models/registry.py` **คอมเมนต์โมเดลที่ไม่ค่อยได้ใช้ออกชั่วคราว** เช่น ถ้าเอาไว้ demo
จริงแค่ EfficientNetB0/MobileNetV2 พอ ตัวใหญ่อย่าง ResNet50V2/Xception ที่กินเวลาต่อภาพเยอะสุด
คอมเมนต์ทิ้งไว้ก่อนได้ (ไม่ต้องลบไฟล์ `.keras` ออก แค่ลบ entry ใน registry ชั่วคราว)

### 2. จำกัดจำนวน thread ที่ TensorFlow ใช้ให้เหมาะกับ CPU จริง
บางครั้ง TF ตั้ง thread เยอะเกินจน context-switch overhead สูงกว่าที่ควร ลองเพิ่ม environment
variable ใน `docker-compose.yml` (ใต้ `dr-screening:`) แล้วปรับตัวเลขตามจำนวน core จริงของเครื่อง:
```yaml
    environment:
      - TF_NUM_INTRAOP_THREADS=4
      - TF_NUM_INTEROP_THREADS=2
```

### 3. อย่าเพิ่ม `--workers` หลายตัวใน uvicorn
อาจดูเหมือนช่วยเรื่อง concurrent request แต่ **แต่ละ worker คือ process แยก ต้องโหลดโมเดลทั้ง 5
ตัวซ้ำอีกชุด** — RAM จะพุ่งเป็นทวีคูณ (5 โมเดล × N worker) เสี่ยง out-of-memory มากกว่าจะเร็วขึ้น
ไม่แนะนำสำหรับเครื่องที่มี RAM จำกัดแบบนี้

### 4. ถ้าจะเร่งจริงจัง — ทางเลือกสุดท้ายคือ GPU
ถ้าในอนาคตมีเครื่องที่มี NVIDIA GPU ให้ใช้ Docker ได้ (ผ่าน NVIDIA Container Toolkit) จะเร็วขึ้นชัดเจน
มาก โดยเฉพาะโหมดเปรียบเทียบ — ถ้าถึงจุดนั้นบอกได้เลย จะทำ `Dockerfile.gpu` แยกให้
