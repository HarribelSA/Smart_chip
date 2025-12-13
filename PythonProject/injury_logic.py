
from blood_detection import detect_blood


def injury_decision(frame, motion_detected):

    blood = detect_blood(frame)

    if blood:
        return "HIGH", "Bleeding or open wounds detected"

    if not motion_detected:
        return "MEDIUM", "No movement detected (possible unconsciousness)"

    return "LOW", "Driver is conscious and moving"
