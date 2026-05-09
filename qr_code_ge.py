import qrcode
import os

# folder to save images
output_dir = "qr_codes"
os.makedirs(output_dir, exist_ok=True)

landmarks = [
    # "pharmacy",
    # "fire_center",
    # "supermarket",
    # "restaurant",
    # "house_1",
    # "house_2",
    # "house_3",
    # "house_4",
    # "house_5",
    "docking_station"
]

custom_payload = {
    "docking_station": "dock",
}

for name in landmarks:
    data = custom_payload.get(name, f"type:{name}")

    qr = qrcode.QRCode(
        version=2,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=2
    )

    qr.add_data(data)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    filename = os.path.join(output_dir, f"{name}.png")
    img.save(filename)

    print(f"Saved {filename}")
