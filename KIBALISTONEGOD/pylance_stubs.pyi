# Stubs pour les modules principaux - Pour corriger les erreurs Pylance

# streamlit stubs
class streamlit:
    @staticmethod
    def set_page_config(**kwargs): pass
    @staticmethod
    def title(text: str): pass
    @staticmethod
    def header(text: str): pass
    @staticmethod
    def subheader(text: str): pass
    @staticmethod
    def write(text): pass
    @staticmethod
    def markdown(text: str): pass
    @staticmethod
    def sidebar(): pass
    @staticmethod
    def columns(spec): pass
    @staticmethod
    def expander(label: str, expanded: bool = False): pass
    @staticmethod
    def checkbox(label: str, value: bool = False, **kwargs): pass
    @staticmethod
    def slider(label: str, min_value=None, max_value=None, value=None, **kwargs): pass
    @staticmethod
    def selectbox(label: str, options, index: int = 0, **kwargs): pass
    @staticmethod
    def color_picker(label: str, value: str, **kwargs): pass
    @staticmethod
    def button(label: str, **kwargs): pass
    @staticmethod
    def spinner(text: str = "In progress..."): pass
    @staticmethod
    def success(text: str): pass
    @staticmethod
    def error(text: str): pass
    @staticmethod
    def warning(text: str): pass
    @staticmethod
    def info(text: str): pass
    @staticmethod
    def image(image, **kwargs): pass
    @staticmethod
    def download_button(label: str, data, file_name: str, mime: str, **kwargs): pass
    @staticmethod
    def file_uploader(label: str, **kwargs): pass
    @staticmethod
    def session_state(): pass

# torch stubs
class torch:
    class nn:
        class Module: pass
        class Conv2d: pass
        class ReLU: pass
        class Sequential: pass
        class Sigmoid: pass
        class MSELoss: pass
        class Linear: pass
        functional: object
    cuda: object
    device: object
    Tensor: object
    @staticmethod
    def from_numpy(array): pass
    @staticmethod
    def zeros(*args, **kwargs): pass
    @staticmethod
    def ones(*args, **kwargs): pass
    @staticmethod
    def randn(*args, **kwargs): pass
    @staticmethod
    def tensor(data, **kwargs): pass
    @staticmethod
    def no_grad(): pass

# PIL stubs
class PIL:
    class Image:
        @staticmethod
        def open(fp): pass
        @staticmethod
        def new(mode: str, size: tuple, color=0): pass
        def save(self, fp, format=None): pass
        def resize(self, size, resample=None): pass
        def convert(self, mode): pass
        def filter(self, filter): pass
        width: int
        height: int
        mode: str
    ImageFilter: object

# numpy stubs
class numpy:
    @staticmethod
    def array(object, dtype=None): pass
    @staticmethod
    def zeros(shape, dtype=None): pass
    @staticmethod
    def ones(shape, dtype=None): pass
    @staticmethod
    def random(): pass
    @staticmethod
    def linspace(start, stop, num=50): pass
    @staticmethod
    def meshgrid(*xi): pass
    @staticmethod
    def clip(a, a_min, a_max): pass
    @staticmethod
    def max(a, axis=None): pass
    @staticmethod
    def min(a, axis=None): pass
    @staticmethod
    def mean(a, axis=None): pass
    @staticmethod
    def std(a, axis=None): pass
    @staticmethod
    def unique(ar): pass
    @staticmethod
    def concatenate(arrays, axis=0): pass
    @staticmethod
    def transpose(a, axes=None): pass
    @staticmethod
    def reshape(a, newshape): pass
    @staticmethod
    def sqrt(x): pass
    @staticmethod
    def sin(x): pass
    @staticmethod
    def cos(x): pass
    @staticmethod
    def arctan2(y, x): pass
    @staticmethod
    def rad2deg(x): pass
    @staticmethod
    def deg2rad(x): pass
    float32: object
    int32: object
    uint8: object

# plotly stubs
class plotly:
    class graph_objects:
        class Figure: pass
        class Scatter: pass
        class Scatter3d: pass
        class Mesh3d: pass
        class Layout: pass
        @staticmethod
        def Histogram(x, **kwargs): pass
        @staticmethod
        def Bar(x, y, **kwargs): pass
    class express:
        @staticmethod
        def scatter(x, y, **kwargs): pass
        @staticmethod
        def scatter_3d(x, y, z, **kwargs): pass
        @staticmethod
        def histogram(x, **kwargs): pass
    class subplots:
        @staticmethod
        def make_subplots(**kwargs): pass

# open3d stubs
class open3d:
    class geometry:
        class PointCloud: pass
        class TriangleMesh: pass
        class LineSet: pass
        class KDTreeFlann: pass
        class KDTreeSearchParamHybrid: pass
        class AxisAlignedBoundingBox: pass
    class io:
        class read_point_cloud: pass
        class write_point_cloud: pass
        class read_triangle_mesh: pass
        class write_triangle_mesh: pass
    class utility:
        class Vector3dVector: pass
        class Vector2iVector: pass
        class DoubleVector: pass
    class visualization:
        class draw_geometries: pass
        class ViewControl: pass
        class RenderOption: pass

# transformers stubs
class transformers:
    class CLIPProcessor: pass
    class CLIPModel: pass
    class AutoTokenizer: pass
    class AutoModel: pass

# sklearn stubs
class sklearn:
    class cluster:
        class KMeans: pass
    class neighbors:
        class NearestNeighbors: pass
    class preprocessing:
        class StandardScaler: pass
    class metrics:
        class accuracy_score: pass

# pandas stubs
class pandas:
    class DataFrame: pass
    class Series: pass
    @staticmethod
    def read_csv(filepath, **kwargs): pass

# psutil stubs
class psutil:
    @staticmethod
    def cpu_percent(interval=None): pass
    @staticmethod
    def virtual_memory(): pass
    @staticmethod
    def disk_usage(path): pass

# pynvml stubs
class pynvml:
    @staticmethod
    def nvmlInit(): pass
    @staticmethod
    def nvmlDeviceGetHandleByIndex(index): pass
    @staticmethod
    def nvmlDeviceGetMemoryInfo(handle): pass
    @staticmethod
    def nvmlDeviceGetUtilizationRates(handle): pass
    @staticmethod
    def nvmlShutdown(): pass

# faiss stubs
class faiss:
    class IndexFlatIP: pass
    class IndexIVFFlat: pass
    @staticmethod
    def read_index(filename): pass
    @staticmethod
    def write_index(index, filename): pass