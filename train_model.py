#!/usr/bin/env python3
"""
Traffic Classification Model Generator
Generates synthetic network flow dataset, trains a RandomForestClassifier,
and exports the trained model to 'traffic_model.onnx'.
"""

import numpy as np

def generate_synthetic_data(num_samples_per_class=500):
    np.random.seed(42)

    X = []
    y = []

    # Target classes:
    # 1: WEB_BROWSING
    # 2: VIDEO_STREAMING
    # 3: VOIP_CALL
    # 4: FILE_TRANSFER

    for _ in range(num_samples_per_class):
        # 1. WEB_BROWSING (Medium size, variable IAT & size)
        mean_pkt = np.random.normal(550, 80)
        std_pkt = np.random.normal(300, 50)
        mean_iat = np.random.normal(80, 25)
        std_iat = np.random.normal(60, 20)
        ratio = np.random.normal(1.2, 0.3)
        p1 = np.random.normal(450, 100)
        p2 = np.random.normal(600, 120)
        p3 = np.random.normal(500, 110)
        p4 = np.random.normal(700, 150)
        p5 = np.random.normal(400, 90)
        X.append([mean_pkt, std_pkt, mean_iat, std_iat, ratio, p1, p2, p3, p4, p5])
        y.append(1)

        # 2. VIDEO_STREAMING (High mean packet size, bursty high frame length)
        mean_pkt = np.random.normal(1150, 90)
        std_pkt = np.random.normal(250, 40)
        mean_iat = np.random.normal(25, 10)
        std_iat = np.random.normal(35, 12)
        ratio = np.random.normal(3.5, 0.8)
        p1 = np.random.normal(1250, 100)
        p2 = np.random.normal(1350, 80)
        p3 = np.random.normal(1100, 120)
        p4 = np.random.normal(1400, 50)
        p5 = np.random.normal(1200, 90)
        X.append([mean_pkt, std_pkt, mean_iat, std_iat, ratio, p1, p2, p3, p4, p5])
        y.append(2)

        # 3. VOIP_CALL (Small packet size, steady fast IAT, low variance)
        mean_pkt = np.random.normal(180, 30)
        std_pkt = np.random.normal(40, 10)
        mean_iat = np.random.normal(20, 5)
        std_iat = np.random.normal(8, 3)
        ratio = np.random.normal(1.0, 0.1)
        p1 = np.random.normal(160, 25)
        p2 = np.random.normal(180, 20)
        p3 = np.random.normal(170, 20)
        p4 = np.random.normal(190, 25)
        p5 = np.random.normal(175, 20)
        X.append([mean_pkt, std_pkt, mean_iat, std_iat, ratio, p1, p2, p3, p4, p5])
        y.append(3)

        # 4. FILE_TRANSFER (Very high packet size, steady bulk stream, low IAT variance)
        mean_pkt = np.random.normal(1380, 50)
        std_pkt = np.random.normal(80, 20)
        mean_iat = np.random.normal(12, 4)
        std_iat = np.random.normal(6, 2)
        ratio = np.random.normal(8.0, 2.0)
        p1 = np.random.normal(1420, 40)
        p2 = np.random.normal(1420, 40)
        p3 = np.random.normal(1420, 40)
        p4 = np.random.normal(1420, 40)
        p5 = np.random.normal(1420, 40)
        X.append([mean_pkt, std_pkt, mean_iat, std_iat, ratio, p1, p2, p3, p4, p5])
        y.append(4)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def main():
    print("[+] Generating synthetic flow dataset...")
    X, y = generate_synthetic_data(600)
    print(f"    Dataset shape: {X.shape}, labels: {y.shape}")

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("[+] Training RandomForestClassifier...")
        clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        clf.fit(X_train, y_train)

        accuracy = clf.score(X_test, y_test)
        print(f"[+] Model Training Accuracy: {accuracy * 100:.2f}%")

        y_pred = clf.predict(X_test)
        target_names = ['WEB_BROWSING', 'VIDEO_STREAMING', 'VOIP_CALL', 'FILE_TRANSFER']
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))

        # Attempt ONNX export via skl2onnx
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType

            initial_type = [('float_input', FloatTensorType([None, 10]))]
            onnx_model = convert_sklearn(clf, initial_types=initial_type)

            with open("traffic_model.onnx", "wb") as f:
                f.write(onnx_model.SerializeToString())
            print("[+] Successfully exported model to 'traffic_model.onnx'")

        except ImportError:
            print("[!] skl2onnx is not installed. To export to ONNX format, run:")
            print("    pip install skl2onnx onnxruntime")

    except ImportError:
        print("[!] scikit-learn is not installed. To train the model, run:")
        print("    pip install scikit-learn numpy")

if __name__ == "__main__":
    main()
