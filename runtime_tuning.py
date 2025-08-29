import platform
import psutil as _ps

def apply_runtime(cfg_runtime):
    try:
        
        p = _ps.Process()
        if hasattr(p, "cpu_affinity") and cfg_runtime.cpu_affinity is not None:
            p.cpu_affinity([int(cfg_runtime.cpu_affinity)])
        pr = str(cfg_runtime.priority).lower()
        if platform.system() == "Windows":
            m = {
                "low": _ps.IDLE_PRIORITY_CLASS,
                "normal": _ps.NORMAL_PRIORITY_CLASS,
                "high": _ps.HIGH_PRIORITY_CLASS,
                "realtime": _ps.REALTIME_PRIORITY_CLASS,
            }
            p.nice(m.get(pr, _ps.NORMAL_PRIORITY_CLASS))
        else:
            nice_map = {"low": 10, "normal": 0, "high": -10, "realtime": -20}
            p.nice(nice_map.get(pr, 0))
    except Exception as e:
        print(f"[runtime] could not apply affinity/priority: {e}")
