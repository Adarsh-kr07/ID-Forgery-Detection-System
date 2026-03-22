
def generate_report(blur, text, regions, cnn_result, metadata):
    report = {
        "blur_score": blur,
        "blur_status": "blurry" if blur < 100 else "clear",
        "cnn_prediction": cnn_result,
        "text_extracted": text[:200],
        "regions_detected": regions,
        "metadata_present": len(metadata) > 0,
        "risk_level": "low"
    }

    # Risk logic
    if blur < 100:
        report["risk_level"] = "medium"

    if cnn_result == "suspicious":
        report["risk_level"] = "high"

    if len(regions) == 0:
        report["risk_level"] = "high"

    return report