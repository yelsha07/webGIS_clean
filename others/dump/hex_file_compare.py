def count_mismatches(file1_path, file2_path):
    mismatches = 0

    with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    max_len = max(len(lines1), len(lines2))

    for i in range(max_len):
        line1 = lines1[i].strip() if i < len(lines1) else "<missing>"
        line2 = lines2[i].strip() if i < len(lines2) else "<missing>"
        line1 = line1.lower()
        line2 = line2.lower()

        if line1 != line2:
            mismatches += 1
            print(f"Line {i+1} mismatch:\n  File1: {line1}\n  File2: {line2}\n")

    print(f"\nTotal mismatches: {mismatches}")
    print(f"File1 total lines: {len(lines1)}")
    print(f"File2 total lines: {len(lines2)}")

count_mismatches("C:\\Users\\student\\Desktop\\HUV\\3_CanSat.hex", "C:\\Users\\student\\Desktop\\HUV\\3_CanSat_UVLE.hex")