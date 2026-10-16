import zipfile
import json
import os

SOURCE = "best_handwriting_model.keras"
OUTPUT = "best_handwriting_model_fixed.keras"


def remove_quantization_config(obj):
    if isinstance(obj, dict):
        obj.pop("quantization_config", None)

        for value in obj.values():
            remove_quantization_config(value)

    elif isinstance(obj, list):
        for item in obj:
            remove_quantization_config(item)


print("Reading model...")

with zipfile.ZipFile(SOURCE, "r") as zin:

    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as zout:

        for item in zin.infolist():

            data = zin.read(item.filename)

            if item.filename == "config.json":
                config = json.loads(data.decode("utf-8"))

                remove_quantization_config(config)

                data = json.dumps(
                    config,
                    separators=(",", ":")
                ).encode("utf-8")

                print("Fixed config.json")

            zout.writestr(item, data)

print()
print("Fixed model created:")
print(OUTPUT)