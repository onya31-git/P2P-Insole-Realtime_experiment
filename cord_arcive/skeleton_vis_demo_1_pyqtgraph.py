import sys
import numpy as np

from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl


class SkeletonViewer(QtWidgets.QWidget):
    def __init__(self, csv_path, parent=None):
        super().__init__(parent)

        # ======== 1. データ読み込み ========
        # 先頭行に "X.1,Y.1,Z.1,..." のヘッダがある前提
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

        # 行: フレーム, 列: 63 (= 21点 × xyz)
        n_frames, n_cols = data.shape
        if n_cols != 63:
            raise ValueError(f"列数が {n_cols} ですが、想定は 63 列です。ヘッダや区切り文字を確認してください。")

        # (frames, 21, 3) に変形
        self.frames = data.reshape(n_frames, 21, 3)

        # ======== 2. pyqtgraph OpenGLビューの準備 ========
        layout = QtWidgets.QVBoxLayout(self)

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor('k')  # 黒背景
        layout.addWidget(self.view)

        self.view.orbit(azim=45, elev=30)  # カメラの向き
        self.view.setCameraPosition(distance=2000)

        # 座標軸（任意）
        axis = gl.GLAxisItem()
        axis.setSize(500, 500, 500)
        self.view.addItem(axis)

        # ======== 3. 点群（関節） ========
        # 最初のフレームを表示
        pos0 = self.frames[0]

        # RGBA (白) で描画
        point_color = np.array([[1.0, 1.0, 1.0, 1.0] for _ in range(pos0.shape[0])])

        self.scatter = gl.GLScatterPlotItem(
            pos=pos0,
            size=10.0,
            color=point_color,
            pxMode=True  # ピクセルサイズ固定
        )
        self.view.addItem(self.scatter)

        # ======== 4. 線分（スケルトン） ========
        # ここでは仮に 1-2-3-...-21 を順番に結ぶ
        # 実際の骨格構造が分かっている場合は、このリストを置き換えてください。
        self.EDGES = [
        (0, 1), (1, 2), (2, 3), (3, 4),                               # 脊椎             # 左右で分けて細かく改行する
        (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (5, 9),  # 手、肘、肩        # 左右で分けて細かく改行する
        (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20), (13, 17)  # 足、腰   # 左右で分けて細かく改行する
    ]

        self.bones = []
        for (i, j) in self.EDGES:
            pts = np.vstack([pos0[i], pos0[j]])
            line = gl.GLLinePlotItem(
                pos=pts,
                mode='lines',
                width=2,
                antialias=True
            )
            self.view.addItem(line)
            self.bones.append(line)

        # ======== 5. アニメーション用タイマー ========
        self.current_frame = 0
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # 約30fps (1000ms / 33 ≒ 30)

    def update_frame(self):
        """次のフレームに進めて描画を更新"""
        self.current_frame = (self.current_frame + 1) % self.frames.shape[0]
        pts = self.frames[self.current_frame]

        # 点群更新
        self.scatter.setData(pos=pts)

        # 線分更新
        for line, (i, j) in zip(self.bones, self.EDGES):
            seg = np.vstack([pts[i], pts[j]])
            line.setData(pos=seg)


def main():
    app = QtWidgets.QApplication(sys.argv)

    # ==== ここでCSVファイル名を指定 ====
    csv_path = "./data/training_data/Skeleton/T005S001_skeleton.csv"  # あなたの csv ファイル名に変更してください

    viewer = SkeletonViewer(csv_path)
    viewer.setWindowTitle("3D Skeleton Viewer (pyqtgraph)")
    viewer.resize(800, 600)
    viewer.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
