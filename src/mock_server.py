import json
import random
import math
from pathlib import Path
from datetime import datetime, timedelta
from uuid import UUID

from fastapi import Body
from shutil import rmtree
import uvicorn
from fastapi import FastAPI, APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
# Base directory for mock data
DATA_DIR = Path(__file__).parent.parent / "data"
HOST = "localhost"
PORT = 8000
origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]

class CustomJSONResponse(JSONResponse):
    def render(self, content: any) -> bytes:
        return json.dumps(content, allow_nan=True, indent=None).encode("utf-8")

def _write_json(path: Path, data: any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

app = FastAPI()
router = APIRouter()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # ["*"] is fine for dev if no credentials are used
    allow_credentials=True,         # set False if not sending cookies/auth headers
    allow_methods=["*"],            # or ["GET","POST","PUT","DELETE","OPTIONS"]
    allow_headers=["*"],            # or explicit header names
)
@router.get("/grids/", response_class=CustomJSONResponse)
async def get_grids(preview: bool = Query(True, description="Include first‐topology preview")):
    """
    Returns list of all grids with optional preview data.
    Reads metadata from data/grids.json and topology previews from data/{grid_id}/.
    """
    grids_file = DATA_DIR / "grids.json"
    if not grids_file.is_file():
        raise HTTPException(status_code=500, detail="grids.json not found in data directory")
    with grids_file.open(encoding="utf-8") as f:
        grids = json.load(f)

    result = []
    for grid in grids:
        grid_id = grid.get("id")
        preview_data = None
        if preview:
            topo_file = DATA_DIR / str(grid_id) / "topologies_preview_True.json"
            if topo_file.is_file():
                with topo_file.open(encoding="utf-8") as tf:
                    topologies = json.load(tf)
                if topologies:
                    preview_data = topologies[0].get("preview")
        result.append({
            "id":          grid_id,
            "name":        grid.get("name"),
            "structure":   grid.get("structure"),
            "preview":     preview_data
        })
    return result

@router.get("/grids/{grid_id}/", response_class=CustomJSONResponse)
async def get_grid_topologies(grid_id: int,
                              preview: bool = Query(True, description="Include topology previews")):
    """
    Returns grid topologies for a given grid.
    Selects from data/{grid_id}/topologies_preview_{preview}.json.
    """
    filename = f"topologies_preview_{preview}.json"
    file_path = DATA_DIR / str(grid_id) / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Topologies file not found")
    with file_path.open(encoding="utf-8") as f:
        return json.load(f)

@router.get("/grids/{grid_id}/{topology_id}/", response_class=CustomJSONResponse)
async def get_grid_topology_detail(grid_id: int, topology_id: str):
    """
    Returns full details of the specified grid topology.
    Reads data from data/{grid_id}/topology_{topology_id}_detail.json.
    """
    file_path = DATA_DIR / str(grid_id) / f"topology_{topology_id}_detail.json"
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Topology detail file not found")
    with file_path.open(encoding="utf-8") as f:
        return json.load(f)

@router.post("/grids/", response_class=CustomJSONResponse)
async def upsert_grid(grid: dict = Body(...)):
    """
    grid: { id: int, name: str, structure: any }
    """
    grids_path = DATA_DIR / "grids.json"
    grids = load_grids()
    # replace or append
    existing = next((g for g in grids if g["id"] == grid["id"]), None)
    if existing:
        existing.update({k: grid[k] for k in ("name","structure")})
    else:
        grids.append({"id": grid["id"], "name": grid["name"], "structure": grid["structure"]})
    _write_json(grids_path, grids)
    return {"status": "ok", "grids": grids}

@router.delete("/grids/{grid_id}/", response_class=CustomJSONResponse)
async def delete_grid(grid_id: int):
    """
    Deletes grid directory and removes entry from grids.json
    """
    # remove directory
    grid_dir = DATA_DIR / str(grid_id)
    if grid_dir.exists():
        rmtree(grid_dir)
    # update grids.json
    grids_path = DATA_DIR / "grids.json"
    grids = load_grids()
    grids = [g for g in grids if g["id"] != grid_id]
    _write_json(grids_path, grids)
    return {"status": "deleted", "grid_id": grid_id}

@router.post("/grids/{grid_id}/{topology_id}/", response_class=CustomJSONResponse)
async def upsert_topology(
    grid_id: int,
    topology_id: UUID,
    topo: dict = Body(...),
):
    """
    topo: full topology JSON (same shape as get_grid_topology_detail)
    """
    # write detail file
    tid = str(topology_id)
    detail_path = DATA_DIR / str(grid_id) / f"topology_{tid}_detail.json"
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(detail_path, topo)

    # update preview list
    preview_path = DATA_DIR / str(grid_id) / "topologies_preview_True.json"
    previews = []
    if preview_path.exists():
        with preview_path.open(encoding="utf-8") as f:
            previews = json.load(f)

    def _get_preview_data(t):
        return {
            "nodes": [{"id": n["idx"], "x": n["x"], "y": n["y"]} for n in t.get("nodes", [])],
            "edges": [
                {"id": e["idx"], "type": e.get("__class__"), "bus1": e["node_id1"], "bus2": e["node_id2"]}
                for e in t.get("edges", [])
            ],
        }
    new_preview = _get_preview_data(topo)

    for entry in previews:
        if str(entry.get("topology_id")) == tid:
            entry["preview"] = new_preview
            entry["n_nodes"] = len(topo.get("nodes", []))
            break
    else:
        previews.append({
            "topology_id": tid,
            "n_nodes": len(topo.get("nodes", [])),
            "preview": new_preview
        })

    _write_json(preview_path, previews)
    return {"status": "ok", "topologies": previews}

@router.delete("/grids/{grid_id}/{topology_id}/", response_class=CustomJSONResponse)
async def delete_topology(grid_id: int, topology_id: UUID):
    """
    Deletes both detail json and removes from preview file
    """
    tid = str(topology_id)
    # delete detail
    detail_path = DATA_DIR / str(grid_id) / f"topology_{tid}_detail.json"
    if detail_path.exists():
        detail_path.unlink()
    # update preview file
    preview_path = DATA_DIR / str(grid_id) / "topologies_preview_True.json"
    if preview_path.exists():
        with preview_path.open(encoding="utf-8") as f:
            existing = json.load(f)
        previews = [e for e in existing if str(e.get("topology_id")) != tid]
        _write_json(preview_path, previews)
    return {"status": "deleted", "topology_id": tid}

# ==== Helpers ==== #

def load_grids():
    file = DATA_DIR / "grids.json"
    if not file.exists():
        raise HTTPException(status_code=500, detail="grids.json missing")
    return json.loads(file.read_text(encoding="utf-8"))

def verify_topology(grid_id: int, topology_id: str):
    topo_file = DATA_DIR / str(grid_id) / "topologies_preview_True.json"
    if not topo_file.exists():
        raise HTTPException(status_code=404, detail="Grid or preview file not found")
    tops = json.loads(topo_file.read_text(encoding="utf-8"))
    for topo in tops:
        if str(topo.get("topology_id")) == str(topology_id):
            return topo.get("preview", {})
    raise HTTPException(status_code=404, detail="Topology not found")

def verify_bus(grid_id: int, topology_id: str, node_id: int):
    preview = verify_topology(grid_id, topology_id)
    node_ids = [n["id"] for n in preview.get("nodes", [])]
    if node_id not in node_ids:
        raise HTTPException(status_code=404, detail=f"Bus {node_id} not in topology")

def verify_line(grid_id: int, topology_id: str, line_id: int):
    preview = verify_topology(grid_id, topology_id)
    edge_ids = [e["id"] for e in preview.get("edges", [])]
    if line_id not in edge_ids:
        raise HTTPException(status_code=404, detail=f"Line {line_id} not in topology")

# ==== State ==== #
_last_voltage = {}
_last_current = {}
_last_power = {}
_last_injection_current = {}
_last_injection_power = {}

# ==== Random Walk ==== #

def random_walk(prev: float, sigma: float, bounds: tuple[float, float]) -> float:
    new = prev + random.gauss(0, sigma)
    low, high = bounds
    return max(low, min(high, new))

# ==== Sampling ==== #

def sample_voltage(grid_id: int, topology_id: str, node_id: int):
    key = (grid_id, topology_id, node_id)
    prev = _last_voltage.get(key, 230.0)
    val = random_walk(prev, 0.5, (210, 250))
    _last_voltage[key] = val
    return {"node_id": node_id, "voltage": round(val, 2)}

def sample_voltage_harmonics(grid_id: int, topology_id: str, node_id: int, orders: list[int]):
    fund = _last_voltage.get((grid_id, topology_id, node_id), 230.0)
    harms = {}
    for n in orders:
        base = fund * (0.01 / max(n, 1))
        harms[n] = round(base * random.uniform(0.8, 1.2), 3)
    return {"node_id": node_id, "harmonics": harms}

def sample_voltage_thd(grid_id: int, topology_id: str, node_id: int, orders: list[int]):
    harms = sample_voltage_harmonics(grid_id, topology_id, node_id, orders)["harmonics"]
    fund = _last_voltage.get((grid_id, topology_id, node_id), 230.0)
    thd = round(math.sqrt(sum(v*v for v in harms.values()))/fund, 4) if fund else 0.0
    return {"node_id": node_id, "thd": thd}

def sample_current(grid_id: int, topology_id: str, line_id: int):
    key = (grid_id, topology_id, line_id)
    prev = _last_current.get(key, 10.0)
    val = random_walk(prev, 0.2, (0, 50))
    _last_current[key] = val
    return {"line_id": line_id, "current": round(val, 3)}

def sample_current_harmonics(grid_id: int, topology_id: str, line_id: int, orders: list[int]):
    fund = _last_current.get((grid_id, topology_id, line_id), 10.0)
    harms = {}
    for n in orders:
        base = fund * (0.02 / max(n, 1))
        harms[n] = round(base * random.uniform(0.8, 1.2), 4)
    return {"line_id": line_id, "harmonics": harms}

def sample_current_thd(grid_id: int, topology_id: str, line_id: int, orders: list[int]):
    harms = sample_current_harmonics(grid_id, topology_id, line_id, orders)["harmonics"]
    fund = _last_current.get((grid_id, topology_id, line_id), 10.0)
    thd = round(math.sqrt(sum(v*v for v in harms.values()))/fund, 4) if fund else 0.0
    return {"line_id": line_id, "thd": thd}

def sample_power(grid_id: int, topology_id: str, line_id: int):
    # sample apparent power in kVA
    key = (grid_id, topology_id, line_id)
    prev_s = _last_power.get(key, 5.0)
    s = random_walk(prev_s, 0.5, (0.1, 10))
    pf = max(0, min(1, random.gauss(0.95, 0.02)))
    p = s * pf  # kW
    q = math.sqrt(max(0, s**2 - p**2))  # kVAr
    _last_power[key] = s
    return {"s_kva": round(s, 3), "p_kw": round(p, 3), "q_kvar": round(q, 3)}

def sample_power_harmonics(grid_id: int, topology_id: str, line_id: int, orders: list[int]):
    key = (grid_id, topology_id, line_id)
    # Retrieve last fundamental apparent power
    prev_s = _last_power.get(key, 5.0)
    s = prev_s
    pf = max(0, min(1, random.gauss(0.95, 0.02)))
    p = s * pf
    q = math.sqrt(max(0, s**2 - p**2))
    harms_s = {n: round(s * (0.005 / max(n, 1)) * random.uniform(0.8, 1.2), 3) for n in orders}
    harms_p = {n: round(p * (0.005 / max(n, 1)) * random.uniform(0.8, 1.2), 3) for n in orders}
    harms_q = {n: round(q * (0.005 / max(n, 1)) * random.uniform(0.8, 1.2), 3) for n in orders}
    return {
        "line_id": line_id,
        "harmonics": {
            "s_kva": harms_s,
            "p_kw": harms_p,
            "q_kvar": harms_q
        }
    }

def sample_power_thd(grid_id: int, topology_id: str, line_id: int, orders: list[int]):
    data = sample_power_harmonics(grid_id, topology_id, line_id, orders)
    harms = data["harmonics"]
    key = (grid_id, topology_id, line_id)
    s = _last_power.get(key, 5.0)
    pf = max(0, min(1, random.gauss(0.95, 0.02)))
    p = s * pf
    q = math.sqrt(max(0, s**2 - p**2))
    thd_s = round(math.sqrt(sum(v*v for v in harms["s_kva"].values()))/s, 4) if s else 0.0
    thd_p = round(math.sqrt(sum(v*v for v in harms["p_kw"].values()))/p, 4) if p else 0.0
    thd_q = round(math.sqrt(sum(v*v for v in harms["q_kvar"].values()))/q, 4) if q else 0.0
    return {
        "line_id": line_id,
        "thd": {
            "s_kva": thd_s,
            "p_kw": thd_p,
            "q_kvar": thd_q
        }
    }


def sample_injection_current(grid_id: int, topology_id: str, node_id: int):
    key = (grid_id, topology_id, node_id)
    prev = _last_injection_current.get(key, 5.0)
    val = random_walk(prev, 0.5, (0, 30))
    _last_injection_current[key] = val
    return {"node_id": node_id, "injection_current": round(val, 3)}

def sample_injection_power(grid_id: int, topology_id: str, node_id: int):
    key = (grid_id, topology_id, node_id)
    prev = _last_injection_power.get(key, 100.0)
    val = random_walk(prev, 20, (-3000, 3000))
    _last_injection_power[key] = val
    return {"node_id": node_id, "injection_power": round(val, 1)}

def sample_sequence(fn, start, end, interval, **kw):
    """

    :param fn:
    :param start: time step (int) or datetime string
    :param end: time step (int) or datetime string
    :param interval: interval in steps or seconds
    :param kw:
    :return:
    """
    seq = []
    # Step-based simulation (integer range)
    if isinstance(start, int) and isinstance(end, int):
        t = start
        while t <= end:
            entry = fn(**kw)
            entry["step"] = t
            seq.append(entry)
            t += interval
    # Time-based simulation (datetime range)
    elif isinstance(start, datetime) and isinstance(end, datetime):
        t = start
        while t <= end:
            entry = fn(**kw)
            entry["timestamp"] = t.isoformat()
            seq.append(entry)
            t += timedelta(seconds=interval)
    else:
        raise ValueError("start and end must both be int or both be datetime")
    return seq

def _parse_start_end(start, end):
    """
    Interpret start/end either as integer steps or ISO datetimes.

    Returns:
        (start_parsed, end_parsed) where both are either int or datetime,
        or (None, None) if either start or end is missing.
    """
    if start is None or end is None:
        return None, None

    # Try integer interpretation first (step-based)
    try:
        s_int = int(start)
        e_int = int(end)
        return s_int, e_int
    except (TypeError, ValueError):
        pass

    # Fallback: interpret as ISO 8601 datetimes (time-based)
    return datetime.fromisoformat(start), datetime.fromisoformat(end)

# ==== Endpoints ==== #

# Voltage
@router.get("/measurements/{grid_id}/{topology_id}/voltage/bus/{node_id}", response_class=CustomJSONResponse)
async def get_voltage(grid_id: int, topology_id: str, node_id: int,
                      start: str = None, end: str = None, interval: int = 60):
    verify_bus(grid_id, topology_id, node_id)
    parsed_start, parsed_end = _parse_start_end(start, end)
    if parsed_start is not None and parsed_end is not None:
        return sample_sequence(
            sample_voltage, parsed_start, parsed_end, interval,
            grid_id=grid_id, topology_id=topology_id, node_id=node_id
        )
    return sample_voltage(grid_id, topology_id, node_id)


@router.get("/measurements/{grid_id}/{topology_id}/voltage/bus/{node_id}/harmonics", response_class=CustomJSONResponse)
async def get_voltage_harmonics(grid_id: int, topology_id: str, node_id: int,
                                orders: str = "1,3,5,7", start: str = None, end: str = None, interval: int = 60):
    verify_bus(grid_id, topology_id, node_id)
    ords = [int(o) for o in orders.split(",")]
    parsed_start, parsed_end = _parse_start_end(start, end)
    if parsed_start is not None and parsed_end is not None:
        return sample_sequence(
            sample_voltage_harmonics, parsed_start, parsed_end, interval,
            grid_id=grid_id, topology_id=topology_id, node_id=node_id, orders=ords
        )
    return sample_voltage_harmonics(grid_id, topology_id, node_id, ords)

@router.get("/measurements/{grid_id}/{topology_id}/voltage/bus/{node_id}/thd", response_class=CustomJSONResponse)
async def get_voltage_thd(grid_id: int, topology_id: str, node_id: int, orders: str = "1,3,5,7"):
    verify_bus(grid_id, topology_id, node_id)
    ords = [int(o) for o in orders.split(",")]
    return sample_voltage_thd(grid_id, topology_id, node_id, ords)

@router.get("/measurements/{grid_id}/{topology_id}/voltage/buses", response_class=CustomJSONResponse)
async def get_all_bus_voltages(grid_id: int, topology_id: str):
    preview = verify_topology(grid_id, topology_id)
    results = []
    for node in preview.get("nodes", []):
        results.append(sample_voltage(grid_id, topology_id, node["id"]))
    return results

# Current
@router.get("/measurements/{grid_id}/{topology_id}/current/line/{line_id}", response_class=CustomJSONResponse)
async def get_line_current(grid_id: int, topology_id: str, line_id: int,
                           start: str = None, end: str = None, interval: int = 60):
    verify_line(grid_id, topology_id, line_id)
    parsed_start, parsed_end = _parse_start_end(start, end)
    if parsed_start is not None and parsed_end is not None:
        return sample_sequence(
            sample_current, parsed_start, parsed_end, interval,
            grid_id=grid_id, topology_id=topology_id, line_id=line_id
        )
    return sample_current(grid_id, topology_id, line_id)


@router.get("/measurements/{grid_id}/{topology_id}/current/line/{line_id}/harmonics", response_class=CustomJSONResponse)
async def get_current_harmonics(grid_id: int, topology_id: str, line_id: int,
                                orders: str = "1,3,5", start: str = None, end: str = None, interval: int = 60):
    verify_line(grid_id, topology_id, line_id)
    ords = [int(o) for o in orders.split(",")]
    parsed_start, parsed_end = _parse_start_end(start, end)
    if parsed_start is not None and parsed_end is not None:
        return sample_sequence(
            sample_current_harmonics, parsed_start, parsed_end, interval,
            grid_id=grid_id, topology_id=topology_id, line_id=line_id, orders=ords
        )
    return sample_current_harmonics(grid_id, topology_id, line_id, ords)

@router.get("/measurements/{grid_id}/{topology_id}/current/line/{line_id}/thd", response_class=CustomJSONResponse)
async def get_current_thd(grid_id: int, topology_id: str, line_id: int, orders: str = "1,3,5"):
    verify_line(grid_id, topology_id, line_id)
    ords = [int(o) for o in orders.split(",")]
    return sample_current_thd(grid_id, topology_id, line_id, ords)

@router.get("/measurements/{grid_id}/{topology_id}/current/lines", response_class=CustomJSONResponse)
async def get_all_line_currents(grid_id: int, topology_id: str):
    preview = verify_topology(grid_id, topology_id)
    results = []
    for edge in preview.get("edges", []):
        results.append(sample_current(grid_id, topology_id, edge["id"]))
    return results

# Power
@router.get("/measurements/{grid_id}/{topology_id}/power/line/{line_id}", response_class=CustomJSONResponse)
async def get_line_power(grid_id: int, topology_id: str, line_id: int,
                         start: str = None, end: str = None, interval: int = 60):
    verify_line(grid_id, topology_id, line_id)
    parsed_start, parsed_end = _parse_start_end(start, end)
    if parsed_start is not None and parsed_end is not None:
        return sample_sequence(
            sample_power, parsed_start, parsed_end, interval,
            grid_id=grid_id, topology_id=topology_id, line_id=line_id
        )
    return sample_power(grid_id, topology_id, line_id)


@router.get("/measurements/{grid_id}/{topology_id}/power/line/{line_id}/harmonics", response_class=CustomJSONResponse)
async def get_power_harmonics(grid_id: int, topology_id: str, line_id: int, orders: str = "1,3,5",
                              start: str = None, end: str = None, interval: int = 60):
    verify_line(grid_id, topology_id, line_id)
    ords = [int(o) for o in orders.split(",")]
    parsed_start, parsed_end = _parse_start_end(start, end)
    if parsed_start is not None and parsed_end is not None:
        return sample_sequence(
            sample_power_harmonics, parsed_start, parsed_end, interval,
            grid_id=grid_id, topology_id=topology_id, line_id=line_id, orders=ords
        )
    return sample_power_harmonics(grid_id, topology_id, line_id, ords)

@router.get("/measurements/{grid_id}/{topology_id}/power/line/{line_id}/thd", response_class=CustomJSONResponse)
async def get_power_thd(grid_id: int, topology_id: str, line_id: int, orders: str = "1,3,5"):
    verify_line(grid_id, topology_id, line_id)
    ords = [int(o) for o in orders.split(",")]
    return sample_power_thd(grid_id, topology_id, line_id, ords)

@router.get("/measurements/{grid_id}/{topology_id}/power/lines", response_class=CustomJSONResponse)
async def get_all_line_powers(grid_id: int, topology_id: str):
    preview = verify_topology(grid_id, topology_id)
    results = []
    for edge in preview.get("edges", []):
        results.append(sample_power(grid_id, topology_id, edge["id"]))
    return results

# Injections
@router.get("/measurements/{grid_id}/{topology_id}/power/injections", response_class=CustomJSONResponse)
async def get_all_injection_powers(grid_id: int, topology_id: str):
    preview = verify_topology(grid_id, topology_id)
    results = []
    for node in preview.get("nodes", []):
        node_id = node["id"]
        results.append(sample_injection_power(grid_id, topology_id, node_id))
    return results

@router.get("/measurements/{grid_id}/{topology_id}/current/injections", response_class=CustomJSONResponse)
async def get_all_injection_currents(grid_id: int, topology_id: str):
    preview = verify_topology(grid_id, topology_id)
    results = []
    for node in preview.get("nodes", []):
        node_id = node["id"]
        results.append(sample_injection_current(grid_id, topology_id, node_id))
    return results

@router.get("/measurements/{grid_id}/{topology_id}/all", response_class=CustomJSONResponse)
async def get_all(grid_id: int, topology_id: str,
                  orders: str = Query("1,3,5,7")):
    preview = verify_topology(grid_id, topology_id)
    ords = [int(o) for o in orders.split(",")]
    nodes = []
    for n in preview.get("nodes", []):
        nid = n["id"]
        v = sample_voltage(grid_id, topology_id, nid)["voltage"]
        h = sample_voltage_harmonics(grid_id, topology_id, nid, ords)["harmonics"]
        nodes.append({"node_id": nid, "voltage": v, "harmonics": h})
    edges = []
    for e in preview.get("edges", []):
        lid = e["id"]
        i_val = sample_current(grid_id, topology_id, lid)["current"]
        pw = sample_power(grid_id, topology_id, lid)
        edges.append({"line_id": lid, "current": i_val, **pw})
    return {"nodes": nodes, "edges": edges}

app.include_router(router)

if __name__ == '__main__':
    uvicorn.run("mock_server:app", host=HOST, port=PORT)
