"""Prepare more inference samples from the 240 unseen videos."""
import json, os, sys, random, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.prepare_inference_data import (
    get_training_ids, get_novel_ids, find_mask_file,
    dilate_mask_video, extract_text_region_frame, vlm_recognize_and_generate,
)
from scripts.prepare_textvace_data import _detect_script, _SCRIPT_FONTS
from scripts.render_glyph_ocr import process_one_sample
import easyocr

infer_dir = 'data/inference_raw'
output_dir = 'data/inference_processed'

train_ids = get_training_ids()
novel_ids = get_novel_ids(infer_dir, train_ids)

with open(os.path.join(output_dir, 'inference_records.json')) as f:
    existing = json.load(f)
existing_ids = {r['id'] for r in existing}

remaining = [v for v in novel_ids if v not in existing_ids]
print(f'Remaining unseen: {len(remaining)}')

random.seed(123)
selected = random.sample(remaining, min(30, len(remaining)))
print(f'Selected: {len(selected)}')

reader = easyocr.Reader(['en', 'ch_sim'], gpu=True, verbose=False)
new_records = []

for i, vid_id in enumerate(selected):
    video_path = os.path.join(infer_dir, 'target_video', vid_id + '.mp4')
    mask_file = find_mask_file(vid_id, os.path.join(infer_dir, 'mask_video'))
    if not mask_file:
        continue

    dilated_path = os.path.join(output_dir, 'dilated_masks', vid_id + '.mp4')
    if not os.path.exists(dilated_path):
        dilate_mask_video(mask_file, dilated_path)

    crop_bytes = extract_text_region_frame(video_path, mask_file)
    if crop_bytes is None:
        continue

    original_text, replacement_text = vlm_recognize_and_generate(crop_bytes)
    if not original_text or not replacement_text:
        continue
    original_text = original_text.strip('"\'')
    replacement_text = replacement_text.strip('"\'')
    if original_text.strip().lower() == replacement_text.strip().lower():
        continue

    script = _detect_script(replacement_text)
    font_path = _SCRIPT_FONTS.get(script, '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf')

    glyph_path = os.path.join(output_dir, 'glyph_videos', vid_id + '.mp4')
    if not os.path.exists(glyph_path):
        ok = process_one_sample(reader, vid_id, video_path, dilated_path,
                                replacement_text, original_text, font_path, glyph_path, 6)

    prompt = f'Change {original_text} to {replacement_text}'
    new_records.append({
        'id': vid_id, 'video_path': video_path, 'mask_path': dilated_path,
        'glyph_path': glyph_path, 'original_text': original_text,
        'replacement_text': replacement_text, 'prompt': prompt,
    })
    if (i + 1) % 5 == 0 or i == 0:
        print(f'  [{i+1}/{len(selected)}] {vid_id}: "{original_text}" -> "{replacement_text}"')

existing.extend(new_records)
with open(os.path.join(output_dir, 'inference_records.json'), 'w') as f:
    json.dump(existing, f, indent=2, ensure_ascii=False)
print(f'Added {len(new_records)}. Total: {len(existing)}')
