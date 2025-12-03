import time
import numpy as np
import open3d as o3d


def load_skeleton_csv(csv_path):
    """
    21ポイント(=63列)の骨格CSVを読み込み、
    shape = (n_frames, 21, 3) の numpy 配列に変換する。
    """
    # 先頭行がヘッダである前提
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

    if data.ndim == 1:
        # 行が1つだけの場合への対処 (1D → 2D)
        data = data[np.newaxis, :]

    n_frames, n_cols = data.shape
    if n_cols != 63:
        raise ValueError(f"列数が {n_cols} 列でした。想定は 63 列(X.1〜Z.41)です。")

    # (frames, 21, 3) に reshape
    frames = data.reshape(n_frames, 21, 3)

    # スケールが大きい場合は適宜調整
    # frames = frames / 10.0

    return frames


def create_skeleton_geometries(initial_points):
    """
    初期フレームの点群から、
    - PointCloud (関節)
    - LineSet (骨格線)
    を作成する。
    """
    # ---- 点群（関節） ----
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(initial_points)

    # 色（白）を一括指定
    colors = np.ones_like(initial_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # ---- 線分（骨格） ----
    # 仮の骨格構造： 1-2-3-...-21 を直線でつなぐ
    # 実際の関節構造が分かっている場合は、このリストを書き換えてください。
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),                               # 脊椎             # 左右で分けて細かく改行する
        (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (5, 9),  # 手、肘、肩        # 左右で分けて細かく改行する
        (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20), (13, 17)  # 足、腰   # 左右で分けて細かく改行する
    ]

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(initial_points)
    lineset.lines = o3d.utility.Vector2iVector(edges)

    # 線の色（青系）を適当に設定
    line_colors = [[0.2, 0.6, 1.0] for _ in edges]
    lineset.colors = o3d.utility.Vector3dVector(line_colors)

    return pcd, lineset, edges


def animate_skeleton_webrtc(frames, fps=30):
    """
    Open3D + WebRTC(WebVisualizer) で骨格をアニメーション表示する。
    frames: shape = (n_frames, 21, 3)
    """
    n_frames = frames.shape[0]
    dt = 1.0 / fps

    # ---- WebRTC サーバを有効化（重要）----
    o3d.visualization.webrtc_server.enable_webrtc()

    # 初期フレーム
    pts0 = frames[0]

    # ジオメトリを作成
    pcd, lineset, edges = create_skeleton_geometries(pts0)

    # Visualizer を作る（通常のネイティブウィンドウ）
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Open3D Skeleton WebVisualizer", width=960, height=720)

    vis.add_geometry(pcd)
    vis.add_geometry(lineset)

    # 視点の調整
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_lookat(pts0.mean(axis=0).tolist())
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(0.3)

    print("WebVisualizer サーバ起動中です。")
    print("ブラウザで http://localhost:8888 を開くと 3D 表示が見られます。")

    frame_idx = 0

    try:
        while True:
            # 表示するフレーム
            pts = frames[frame_idx]

            # 点群の更新
            pcd.points = o3d.utility.Vector3dVector(pts)

            # LineSet の点も更新
            lineset.points = o3d.utility.Vector3dVector(pts)

            # Visualizer に更新を反映
            vis.update_geometry(pcd)
            vis.update_geometry(lineset)

            vis.poll_events()
            vis.update_renderer()

            time.sleep(dt)

            frame_idx = (frame_idx + 1) % n_frames
    except KeyboardInterrupt:
        # Ctrl+C で終了
        pass
    finally:
        vis.destroy_window()


def main():
    # あなたの CSV ファイルパスに変更してください
    csv_path = "./data/training_data/Skeleton/T005S001_skeleton.csv"

    frames = load_skeleton_csv(csv_path)
    print(f"Frames: {frames.shape[0]}, Points: {frames.shape[1]}")

    animate_skeleton_webrtc(frames, fps=30)


if __name__ == "__main__":
    main()
