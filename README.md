# Smart Gallery Similarity Engine

نسخة Production نظيفة لتجميع الصور المتشابهة. تحتوي فقط على الـpipeline النهائي
والـIncremental Processing، بدون California-ND، التجارب، المعايرة، الاختبارات،
أو نتائج التشغيل القديمة. يمكن توليد مجلدات معاينة للصور عند الطلب.

## الـpipeline النهائي

```text
HSV color + fixed 24-hour capture-time distance
→ Affinity Propagation initial partition
→ SIFT + bidirectional matching + SMR
→ combined similarity with weights 3:2:1
→ average-linkage grouping at threshold 0.579
```

أول تشغيل لمعرض معيّن ينفذ Full Build. التشغيلات اللاحقة تكتشف
Added / Modified / Deleted / Unchanged وتستخدم Incremental Processing تلقائيًا.
إذا كان التغيير كبيرًا أو غير آمن، ينتقل المحرك تلقائيًا إلى Full Rebuild آمن.

## المتطلبات

- Python 3.10
- الصيغ المدعومة: JPG, JPEG, PNG, WEBP
- صورتان على الأقل في أول Full Build
- يفضّل تزويد كل صورة بـ`asset_id` ثابت ووقت التقاط حقيقي

## التثبيت على Windows PowerShell

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

## التشغيل

```powershell
smart-gallery analyze `
  --gallery-id user_123 `
  --input-dir "D:\Photos\Gallery" `
  --runtime-dir ".\runtime"
```

النتائج تحفظ تلقائيًا هنا:

```text
runtime/galleries/user_123/groups.json
runtime/galleries/user_123/last_run.json
```

لتوليد مجلد مستقل لكل مجموعة مع نسخ الصور داخله، أضيفي `--previews`:

```powershell
smart-gallery analyze `
  --gallery-id user_123 `
  --input-dir "D:\Photos\Gallery" `
  --runtime-dir ".\runtime" `
  --previews
```

ستظهر المعاينات هنا، وتُعاد مزامنتها بالكامل في كل تشغيل يستخدم الخيار:

```text
runtime/galleries/user_123/previews/
├── group_0001/
│   ├── 001_IMG_1001.jpg
│   └── 002_IMG_1002.jpg
└── group_0002/
    ├── 001_IMG_2050.jpg
    └── 002_IMG_2051.jpg
```

المعاينات ليست جزءًا من الخوارزمية أو الـCache، ويمكن حذف مجلد `previews`
في أي وقت من دون التأثير في الـState أو الـIncremental Processing.

السلوك المتوقع:

- لا توجد State سابقة: `full_build`
- تغييرات صغيرة: `incremental_update`
- لا تغييرات: `no_changes`
- تغيير كبير/غير آمن: `full_rebuild`

لإجبار Full Rebuild آمن مع الاحتفاظ بالـState القديمة حتى ينجح البناء الجديد:

```powershell
smart-gallery analyze `
  --gallery-id user_123 `
  --input-dir "D:\Photos\Gallery" `
  --runtime-dir ".\runtime" `
  --fresh
```

لطباعة النتيجة كاملة كـJSON:

```powershell
smart-gallery analyze --gallery-id user_123 --input-dir "D:\Photos\Gallery" --json
```

## Manifest اختياري

في تطبيق الموبايل يجب إرسال `asset_id` و`capture_time`. للاختبار على مجلد محلي:

```json
[
  {
    "filename": "IMG_1001.jpg",
    "asset_id": "phone_asset_1001",
    "capture_time": "2026-08-13T10:15:20"
  },
  {
    "filename": "IMG_1002.jpg",
    "asset_id": "phone_asset_1002",
    "capture_time": "2026-08-13T10:15:23"
  }
]
```

ثم:

```powershell
smart-gallery analyze `
  --gallery-id user_123 `
  --input-dir "D:\Photos\Gallery" `
  --manifest "D:\Photos\Gallery\manifest.json"
```

بدون Manifest، يولد النظام `asset_id` من المسار النسبي ويقرأ الوقت من EXIF.
إذا لم يوجد EXIF، يستخدم file modification time ويصدر Warning.

## شكل النتيجة

`groups.json` يحتوي مجموعات `asset_id` فقط:

```json
[
  ["phone_asset_1001", "phone_asset_1002"],
  ["phone_asset_2050", "phone_asset_2051", "phone_asset_2052"]
]
```

أما `last_run.json` فيحتوي معلومات كاملة مثل mode، group IDs، representative،
عدد الصور، التحذيرات، وإحصاءات العمل المنفذ.

## Runtime State

```text
runtime/
└── galleries/
    └── user_123/
        ├── state.sqlite3
        ├── groups.json
        ├── last_run.json
        ├── previews/              # عند استخدام --previews
        └── features/
            └── <opaque-cache-files>.npz
```

هذه الملفات Derived State ضرورية للـIncremental Processing. لا تحذفها بين
التشغيلات العادية. الصور الأصلية لا تُنسخ إلى Runtime State.

لبداية نظيفة كليًا لمعرض ما، احذف مجلده فقط:

```powershell
Remove-Item ".\runtime\galleries\user_123" -Recurse -Force
```

ثم التشغيل التالي يصبح Full Build جديدًا.

## Python API للـFastAPI أو الباك اند

```python
from pathlib import Path
from smart_gallery_similarity import analyze_gallery

result = analyze_gallery(
    input_dir=Path(r"D:\Photos\Gallery"),
    gallery_id="user_123",
    runtime_dir=Path("runtime"),
    manifest_path=Path(r"D:\Photos\Gallery\manifest.json"),
)

payload = result.to_dict()
```

يمكن للباك اند إرجاع `payload["groups"]` إلى الموبايل، ثم حذف الصور المؤقتة
المرفوعة. يجب الاحتفاظ بـ`runtime/galleries/<gallery_id>` للتحديثات اللاحقة.

## الإعدادات المثبتة

- Time policy: fixed horizon = 24 hours
- SIFT maximum edge: 1024 px
- Combined weights: SIFT 3, Color 2, Time 1
- Final linkage: average
- Final similarity threshold: 0.579
- Minimum group size: 2

Quality rating:
https://github.com/farahmikari/ImageEvaluationProject.git

Face Grouping:
https://github.com/Ahmad-Jouhar-1/Face-Grouping.git

Similar:
https://github.com/Ayaob2004/similarity-service.git

Backend:
https://github.com/reham270/image_eval_backend.git

Flutter:
https://github.com/Ahmad-Jouhar-1/Smart-Gallery.git
