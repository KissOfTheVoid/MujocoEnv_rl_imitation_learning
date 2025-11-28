from PySide6 import QtWidgets
from rendercanvas.qt import QRenderWidget  # новый canvas
import wgpu   # просто чтобы модуль был загружен
import pygfx as gfx


class GfxWidget(QRenderWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # контекст для WebGPU
        present_context = self.get_wgpu_context()

        # рендерер pygfx поверх этого контекста
        self.renderer = gfx.renderers.WgpuRenderer(present_context)

        # дальше — как раньше: создаёшь сцену, камеру и т.п.
        self.scene = gfx.Scene()
        self.camera = gfx.PerspectiveCamera(70, 16/9)
        self.camera.position.z = 400

        cube = gfx.Mesh(
            gfx.box_geometry(200, 200, 200),
            gfx.MeshPhongMaterial(color="#336699"),
        )
        self.scene.add(cube)

        @self.request_draw
        def draw():
            self.renderer.render(self.scene, self.camera)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    w = GfxWidget()
    w.resize(800, 600)
    w.show()
    app.exec()
