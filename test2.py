import numpy as np
import sys

# Force UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

def analyze_C3_only():
    print("=== 分析鑽石晶格的 8 個 C3 旋轉 + 1 個 Identity ===")

    # 1. 定義 4 個鍵 (Bonds)
    # 這些是四面體的四個頂點方向 (normalized)
    d = [
        np.array([1., 1., 1.]),     # Bond 0
        np.array([1., -1., -1.]),   # Bond 1
        np.array([-1., 1., -1.]),   # Bond 2
        np.array([-1., -1., 1.])    # Bond 3
    ]
    # 正規化方便計算
    d = [v / np.linalg.norm(v) for v in d]

    # 2. 定義旋轉矩陣生成器 (Rodrigues' formula)
    def get_rot_mat(axis, theta):
        axis = axis / np.linalg.norm(axis)
        a, b, c = axis
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        return np.array([
            [cos_t + a**2*(1-cos_t),   a*b*(1-cos_t) - c*sin_t, a*c*(1-cos_t) + b*sin_t],
            [b*a*(1-cos_t) + c*sin_t, cos_t + b**2*(1-cos_t),   b*c*(1-cos_t) - a*sin_t],
            [c*a*(1-cos_t) - b*sin_t, c*b*(1-cos_t) + a*sin_t, cos_t + c**2*(1-cos_t)]
        ])

    # 3. 建立 9 個操作 (1 Identity + 4 axes * 2 rotations)
    operations = []
    
    # (A) Identity
    operations.append(("Identity", np.eye(3)))
    
    # (B) 8 個 C3 旋轉 (繞著 4 個鍵轉 +/- 120度)
    angles = [2*np.pi/3, -2*np.pi/3] # 120, 240 (or -120)
    
    for i in range(4): # 繞著第 i 號鍵轉
        axis = d[i]
        for angle in angles:
            R = get_rot_mat(axis, angle)
            name = f"C3 (Axis {i}, {int(np.degrees(angle))} deg)"
            operations.append((name, R))

    # 4. 計算排列矩陣與特徵值
    np.set_printoptions(precision=2, suppress=True)
    
    for name, R in operations:
        print(f"\n--- 操作: {name} ---")
        
        # 建立 4x4 排列矩陣 P
        # P_ij = 1 代表: 旋轉後的第 j 號鍵 變成了 第 i 號鍵
        P = np.zeros((4, 4), dtype=complex)
        
        # 測試每個鍵旋轉後變成了哪個鍵
        mapping = []
        for j in range(4): # 原本的鍵 j
            rotated_vec = R @ d[j]
            
            # 找它對應到哪一個新的鍵 i
            found = False
            for i in range(4):
                if np.allclose(rotated_vec, d[i], atol=1e-4):
                    P[i, j] = 1.0
                    mapping.append(f"{j}->{i}")
                    found = True
                    break
            if not found:
                print(f"  警告: 鍵 {j} 旋轉後沒有對應到任何鍵！(可能軸選錯了)")

        print(f"  排列映射: {', '.join(mapping)}")
        print(f"  排列矩陣 P:\n{np.real(P)}") # 這裡只印實部因為排列矩陣由0和1組成
        
        # 計算特徵值
        eigs = np.linalg.eigvals(P)
        
        # 整理特徵值顯示 (排序並標註 omega)
        # omega = exp(i 2pi/3) = -0.5 + 0.866j
        print("  特徵值 (Eigenvalues):")
        for e in eigs:
            # 判斷它是不是 1, omega, 或 omega^2
            note = ""
            if np.isclose(e, 1): note = "(1)"
            elif np.isclose(e, -0.5 + 0.8660254j): note = "(ω)"
            elif np.isclose(e, -0.5 - 0.8660254j): note = "(ω^2)"
            
            print(f"    {e:.2f}  {note}")

if __name__ == "__main__":
    analyze_C3_only()




