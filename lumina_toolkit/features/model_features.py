"""Features that require a learned model. Pipeline glue lives here;
the model itself is supplied via `backends.register_backend`."""
from ..backends import require_backend
from ..registry import feature


@feature("face_swap", "Face Editor", "Swap a face from `source` onto `target`.", needs_backend="face_swap")
def face_swap(source, target):
    return require_backend("face_swap")(source, target)


@feature("face_repair", "Face Repair", "Restore facial detail.", needs_backend="face_restore")
def face_repair(img):
    return require_backend("face_restore")(img)


@feature("ocr", "Picture to Text", "Extract text from an image.", needs_backend="ocr")
def ocr(img):
    return require_backend("ocr")(img)


@feature("tts", "Text to Audio", "Synthesize speech from text.", needs_backend="tts")
def tts(text, voice=None):
    return require_backend("tts")(text, voice)


@feature("stt", "Speech to Text", "Transcribe audio.", needs_backend="stt")
def stt(audio_path):
    return require_backend("stt")(audio_path)


@feature("chat", "AI Chat", "Conversational assistant.", needs_backend="chat")
def chat(messages):
    return require_backend("chat")(messages)


@feature("translate", "Translation", "Translate text between languages.", needs_backend="translate")
def translate(text, target_lang, source_lang=None):
    return require_backend("translate")(text, target_lang, source_lang)


@feature("video_bg", "Video Background Changer", "Replace video background.", needs_backend="video_matting")
def video_bg(video_path, background, out_path):
    return require_backend("video_matting")(video_path, background, out_path)


@feature("presenter", "Custom presenter", "Custom voice/avatar presenter.", needs_backend="presenter")
def presenter(text, profile):
    return require_backend("presenter")(text, profile)
