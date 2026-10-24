# Pure-Python decision logic for the disease API (no torch needed here).
# Takes the 27 raw model scores, keeps only the crop the farmer picked,
# and turns them into probabilities, top guesses and a plain-English message.
import math

NOT_SURE_MESSAGE = (
    "I am not sure what this is. Please take a clearer photo: daylight, "
    "close to one leaf, plain background, and keep the leaf in focus."
)

# Extra honesty note shown with certain crops (the app can translate it later).
CROP_NOTES = {
    "Onion": (
        "Some viral and root diseases cannot always be confirmed from a leaf photo. "
        "Please also show the plant to your local Krishi Vigyan Kendra (KVK)."
    ),
}


def softmax(values):
    biggest = max(values)
    exps = [math.exp(v - biggest) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def split_class_name(class_name):
    crop, disease = class_name.split("___", 1)
    return crop, disease.replace("_", " ")


def predict_for_crop(logits, class_names, crop, threshold=0.7, top_k=2, precautions=None):
    wanted = crop.strip().lower()
    idx = [i for i, name in enumerate(class_names) if name.split("___")[0].lower() == wanted]
    if not idx:
        raise ValueError(f"Unknown crop '{crop}'")

    probs = softmax([logits[i] for i in idx])          # only this crop's classes compete
    ranked = sorted(zip(idx, probs), key=lambda pair: pair[1], reverse=True)[:top_k]

    top = []
    for i, p in ranked:
        _, disease = split_class_name(class_names[i])
        top.append({
            "class": class_names[i],
            "disease": disease,
            "probability": round(p, 4),
            "is_healthy": disease.lower() == "healthy",
        })

    crop_name = class_names[idx[0]].split("___")[0]
    best = top[0]
    advice = []

    if best["probability"] < threshold:
        status, message = "not_sure", NOT_SURE_MESSAGE
    elif best["is_healthy"]:
        status, message = "ok", "The leaf looks healthy."
    else:
        status = "ok"
        message = f"The most likely problem is {best['disease']}."
        if len(top) > 1 and top[1]["probability"] >= 0.25:
            second = "healthy" if top[1]["is_healthy"] else top[1]["disease"]
            message += f" It could also be {second}."
        if precautions:
            advice = precautions.get(best["class"], [])

    return {
        "crop": crop_name,
        "status": status,
        "message": message,
        "top": top,
        "threshold": threshold,
        "note": CROP_NOTES.get(crop_name, ""),
        "precautions": advice,
    }
