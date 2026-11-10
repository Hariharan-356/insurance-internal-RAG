from pathlib import Path


def make_pdf(path: Path, lines):
    escaped = []
    for line in lines:
        s = line.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
        escaped.append(s)

    stream_text = "BT\n/F1 12 Tf\n50 730 Td\n"
    for i, s in enumerate(escaped):
        stream_text += f"({s}) Tj\n"
        if i != len(escaped) - 1:
            stream_text += "0 -22 Td\n"
    stream_text += "ET\n"

    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>\nendobj\n",
    ]

    stream_bytes = stream_text.encode('latin-1')
    objects.append(f"4 0 obj\n<< /Length {len(stream_bytes)} >>\nstream\n".encode('latin-1') + stream_bytes + b"endstream\nendobj\n")
    objects.append(b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf.extend(obj)

    xref_pos = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode('latin-1'))
    pdf.extend(b"0000000000 65535 f \n")
    for i in range(1, len(objects) + 1):
        pdf.extend(f"{offsets[i]:010d} 00000 n \n".encode('latin-1'))
    pdf.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode('latin-1'))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(pdf))
    print(f"Wrote {path} ({path.stat().st_size} bytes)")


if __name__ == '__main__':
    make_pdf(
        Path(r"C:\Users\HariharanMurugesan\OneDrive - ValueMomentum, Inc\Documents\insurance_kb_assistant\insurance_kb_assistant\data\documents\internal\claims_approval_policy.pdf"),
        [
            "Claims approval limits.",
            "Claims up to 50,000 can be approved by a Claims Executive.",
            "Claims between 50,001 and 500,000 require Senior Adjuster approval.",
            "Claims above 500,000 require executive review.",
        ],
    )
