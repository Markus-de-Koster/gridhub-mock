import json
import random
import math
from pathlib import Path
from datetime import datetime, timedelta
from fastapi import Body
from shutil import rmtree
import uvicorn
from fastapi import FastAPI, APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
# Base directory for mock data
DATA_DIR = Path(__file__).parent.parent / "data"
HOST = "localhost"
PORT = 8000

class CustomJSONResponse(JSONResponse):
    def render(self, content: any) -> bytes:
        return json.dumps(content, allow_nan=True).encode("utf-8")

def _write_json(path: Path, data: any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
app = FastAPI()
router = APIRouter()

@router.get("/grids/", response_class=CustomJSONResponse)
async def get_grids(preview: bool = Query(True, description="Include first‐topology preview")):
    """
    Returns list of all grids with optional preview data.
    Reads metadata from data/grids.json and topology previews from data/{grid_id}/.
    """
    grids_file = DATA_DIR / "grids.json"
    if not grids_file.is_file():
        raise HTTPException(status_code=500, detail="grids.json not found in data directory")
    with grids_file.open() as f:
        grids = json.load(f)

    result = []
    for grid in grids:
        grid_id = grid.get("id")
        preview_data = None
        if preview:
            topo_file = DATA_DIR / str(grid_id) / "topologies_preview_True.json"
            if topo_file.is_file():
                with topo_file.open() as tf:
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
    with file_path.open() as f:
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
    with file_path.open() as f:
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
    topology_id: int,
    topo: dict = Body(...)
):
    """
    topo: full topology JSON (same shape as get_grid_topology_detail)
    """
    # write detail file
    detail_path = DATA_DIR / str(grid_id) / f"topology_{topology_id}_detail.json"
    _write_json(detail_path, topo)

    # update preview list
    preview_path = DATA_DIR / str(grid_id) / "topologies_preview_True.json"
    previews = []
    if preview_path.exists():
        previews = json.loads(preview_path.read_text())
    # regenerate this entry’s preview
    def _get_preview_data(t):
        return {
            "nodes": [{"id": n["idx"], "x": n["x"], "y": n["y"]} for n in t["nodes"]],
            "edges": [
                {"id": e["idx"], "type": e["__class__"], "bus1": e["node_id1"], "bus2": e["node_id2"]}
                for e in t["edges"]
            ],
        }
    new_preview = _get_preview_data(topo)
    # find or append
    for entry in previews:
        if entry["topology_id"] == topology_id:
            entry["preview"] = new_preview
            break
    else:
        previews.append({"topology_id": topology_id,
                         "n_nodes": len(topo["nodes"]),
                         "preview": new_preview})
    _write_json(preview_path, previews)

    return {"status": "ok", "topologies": previews}

@router.delete("/grids/{grid_id}/{topology_id}/", response_class=CustomJSONResponse)
async def delete_topology(grid_id: int, topology_id: int):
    """
    Deletes both detail json and removes from preview file
    """
    # delete detail
    detail_path = DATA_DIR / str(grid_id) / f"topology_{topology_id}_detail.json"
    if detail_path.exists():
        detail_path.unlink()
    # update preview file
    preview_path = DATA_DIR / str(grid_id) / "topologies_preview_True.json"
    if preview_path.exists():
        previews = [e for e in json.loads(preview_path.read_text())
                    if e["topology_id"] != topology_id]
        _write_json(preview_path, previews)
    return {"status": "deleted", "topology_id": topology_id}

# ==== Helpers ==== #

def load_grids():
    file = DATA_DIR / "grids.json"
    if not file.exists():
        raise HTTPException(status_code=500, detail="grids.json missing")
    return json.loads(file.read_text())


def verify_topology(grid_id: int, topology_id: str):
    topo_file = DATA_DIR / str(grid_id) / "topologies_preview_True.json"
    if not topo_file.exists():
        raise HTTPException(status_code=404, detail="Grid or preview file not found")
    tops = json.loads(topo_file.read_text())
    for topo in tops:
        if topo.get("topology_id") == topology_id:
            # preview dict contains 'structure' preview nodes/edges
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

def sample_bus_voltage(grid_id: int, topology_id: str, node_id: int):
    key = (grid_id, topology_id, node_id)
    prev = _last_voltage.get(key, 230.0)
    val = random_walk(prev, 0.5, (210, 250))
    _last_voltage[key] = val
    return {"node_id": node_id, "voltage": round(val, 2)}


def sample_bus_harmonics(grid_id: int, topology_id: str, node_id: int, orders: list[int]):
    fund = _last_voltage.get((grid_id, topology_id, node_id), 230.0)
    harms = {}
    for n in orders:
        base = fund * (0.01 / n)
        harms[n] = round(base * random.uniform(0.8,1.2),3)
    return {"node_id": node_id, "harmonics": harms}


def sample_bus_thd(grid_id: int, topology_id: str, node_id: int, orders: list[int]):
    harms = sample_bus_harmonics(grid_id, topology_id, node_id, orders)["harmonics"]
    fund = _last_voltage.get((grid_id, topology_id, node_id), 230.0)
    thd = round(math.sqrt(sum(v*v for v in harms.values()))/fund,4)
    return {"node_id": node_id, "thd": thd}


def sample_line_current(grid_id: int, topology_id: str, line_id: int):
    key = (grid_id, topology_id, line_id)
    prev = _last_current.get(key, 10.0)
    val = random_walk(prev, 0.2, (0, 50))
    _last_current[key] = val
    return {"line_id": line_id, "current": round(val,3)}


def sample_current_harmonics(grid_id: int, topology_id: str, line_id: int, orders: list[int]):
    fund = _last_current.get((grid_id, topology_id, line_id), 10.0)
    harms = {}
    for n in orders:
        base = fund*(0.02/n)
        harms[n] = round(base*random.uniform(0.8,1.2),4)
    return {"line_id": line_id, "harmonics": harms}


def sample_current_thd(grid_id: int, topology_id: str, line_id: int, orders: list[int]):
    harms = sample_current_harmonics(grid_id, topology_id, line_id, orders)["harmonics"]
    fund = _last_current.get((grid_id, topology_id, line_id),10.0)
    thd = round(math.sqrt(sum(v*v for v in harms.values()))/fund,4)
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


def sample_power_harmonics(grid_id: int, topology_id: int, line_id: int, orders: list[int]):
    key = (grid_id, topology_id, line_id)
    # Retrieve last fundamental apparent power
    prev_s = _last_power.get(key, 5.0)
    # Derive fundamental p and q like in sample_power
    s = prev_s
    pf = max(0, min(1, random.gauss(0.95, 0.02)))
    p = s * pf
    q = math.sqrt(max(0, s**2 - p**2))
    # Generate harmonics for each quantity
    harms_s = {n: round(s * (0.005 / n) * random.uniform(0.8, 1.2), 3) for n in orders}
    harms_p = {n: round(p * (0.005 / n) * random.uniform(0.8, 1.2), 3) for n in orders}
    harms_q = {n: round(q * (0.005 / n) * random.uniform(0.8, 1.2), 3) for n in orders}
    return {
        "line_id": line_id,
        "harmonics": {
            "s_kva": harms_s,
            "p_kw": harms_p,
            "q_kvar": harms_q
        }
    }


def sample_power_thd(grid_id: int, topology_id: int, line_id: int, orders: list[int]):
    # Retrieve harmonics
    data = sample_power_harmonics(grid_id, topology_id, line_id, orders)
    harms = data["harmonics"]
    key = (grid_id, topology_id, line_id)
    # Fundamental values
    s = _last_power.get(key, 5.0)
    pf = max(0, min(1, random.gauss(0.95, 0.02)))
    p = s * pf
    q = math.sqrt(max(0, s**2 - p**2))
    # Compute THD per quantity
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
    key=(grid_id, topology_id, node_id)
    prev=_last_injection_current.get(key,5.0)
    val=random_walk(prev,0.5,(0,30))
    _last_injection_current[key]=val
    return {"node_id":node_id, "injection_current":round(val,3)}


def sample_injection_power(grid_id: int, topology_id: str, node_id: int):
    key=(grid_id, topology_id, node_id)
    prev=_last_injection_power.get(key,100.0)
    val=random_walk(prev,20,(-3000,3000))
    _last_injection_power[key]=val
    return {"node_id":node_id, "injection_power":round(val,1)}


def sample_sequence(fn,start,end,interval,**kw):
    t=start; seq=[]
    while t<=end:
        entry=fn(**kw)
        entry["timestamp"]=t.isoformat()
        seq.append(entry)
        t+=timedelta(seconds=interval)
    return seq

# ==== Endpoints ==== #

# Voltage
@router.get("/measurements/{grid_id}/{topology_id}/voltage/bus/{node_id}", response_class=CustomJSONResponse)
async def get_voltage(grid_id:int, topology_id:str, node_id:int,
                      start:str=None,end:str=None,interval:int=60):
    verify_bus(grid_id, topology_id, node_id)
    if start and end:
        return sample_sequence(sample_bus_voltage, datetime.fromisoformat(start), datetime.fromisoformat(end), interval,
                                grid_id=grid_id, topology_id=topology_id, node_id=node_id)
    return sample_bus_voltage(grid_id, topology_id, node_id)

@router.get("/measurements/{grid_id}/{topology_id}/voltage/bus/{node_id}/harmonics", response_class=CustomJSONResponse)
async def get_voltage_harmonics(grid_id:int, topology_id:str, node_id:int,
                                 orders:str="1,3,5,7", start:str=None,end:str=None, interval:int=60):
    verify_bus(grid_id, topology_id, node_id)
    ords=[int(o) for o in orders.split(",")]
    if start and end:
        return sample_sequence(sample_bus_harmonics, datetime.fromisoformat(start), datetime.fromisoformat(end), interval,
                                grid_id=grid_id, topology_id=topology_id, node_id=node_id, orders=ords)
    return sample_bus_harmonics(grid_id, topology_id, node_id, ords)

@router.get("/measurements/{grid_id}/{topology_id}/voltage/bus/{node_id}/thd", response_class=CustomJSONResponse)
async def get_voltage_thd(grid_id:int, topology_id:str, node_id:int, orders:str="1,3,5,7"):
    verify_bus(grid_id, topology_id, node_id)
    ords=[int(o) for o in orders.split(",")]
    return sample_bus_thd(grid_id,topology_id,node_id,ords)

@router.get("/measurements/{grid_id}/{topology_id}/voltage/buses", response_class=CustomJSONResponse)
async def get_all_bus_voltages(grid_id:int, topology_id:str):
    preview = verify_topology(grid_id,topology_id)
    results=[]
    for node in preview.get("nodes",[]):
        results.append(sample_bus_voltage(grid_id, topology_id, node["id"]))
    return results

# Current
@router.get("/measurements/{grid_id}/{topology_id}/current/line/{line_id}", response_class=CustomJSONResponse)
async def get_line_current(grid_id:int, topology_id:str, line_id:int, start:str=None,end:str=None,interval:int=60):
    verify_line(grid_id,topology_id,line_id)
    if start and end:
        return sample_sequence(sample_line_current, datetime.fromisoformat(start), datetime.fromisoformat(end), interval,
                                grid_id=grid_id, topology_id=topology_id, line_id=line_id)
    return sample_line_current(grid_id,topology_id,line_id)

@router.get("/measurements/{grid_id}/{topology_id}/current/line/{line_id}/harmonics", response_class=CustomJSONResponse)
async def get_current_harmonics(grid_id:int, topology_id:str, line_id:int,
                                orders:str="1,3,5",start:str=None,end:str=None,interval:int=60):
    verify_line(grid_id,topology_id,line_id)
    ords=[int(o) for o in orders.split(",")]
    if start and end:
        return sample_sequence(sample_current_harmonics, datetime.fromisoformat(start), datetime.fromisoformat(end), interval,
                                grid_id=grid_id, topology_id=topology_id, line_id=line_id, orders=ords)
    return sample_current_harmonics(grid_id,topology_id,line_id,ords)

@router.get("/measurements/{grid_id}/{topology_id}/current/line/{line_id}/thd", response_class=CustomJSONResponse)
async def get_current_thd(grid_id:int, topology_id:str,line_id:int, orders:str="1,3,5"):
    verify_line(grid_id,topology_id,line_id)
    ords=[int(o) for o in orders.split(",")]
    return sample_current_thd(grid_id,topology_id,line_id,ords)

@router.get("/measurements/{grid_id}/{topology_id}/current/lines", response_class=CustomJSONResponse)
async def get_all_line_currents(grid_id:int, topology_id:str):
    preview=verify_topology(grid_id,topology_id)
    results=[]
    for edge in preview.get("edges",[]):
        results.append(sample_line_current(grid_id,topology_id,edge["id"]))
    return results

# Power
@router.get("/measurements/{grid_id}/{topology_id}/power/line/{line_id}", response_class=CustomJSONResponse)
async def get_line_power(grid_id:int, topology_id:str,line_id:int,start:str=None,end:str=None,interval:int=60):
    verify_line(grid_id,topology_id,line_id)
    if start and end:
        return sample_sequence(sample_power, datetime.fromisoformat(start), datetime.fromisoformat(end), interval,
                                grid_id=grid_id, topology_id=topology_id, line_id=line_id)
    return sample_power(grid_id,topology_id,line_id)

@router.get("/measurements/{grid_id}/{topology_id}/power/line/{line_id}/harmonics", response_class=CustomJSONResponse)
async def get_power_harmonics(grid_id:int, topology_id:str,line_id:int,orders:str="1,3,5",start:str=None,end:str=None,interval:int=60):
    verify_line(grid_id,topology_id,line_id)
    ords=[int(o) for o in orders.split(",")]
    if start and end:
        return sample_sequence(sample_power_harmonics, datetime.fromisoformat(start), datetime.fromisoformat(end), interval,
                                grid_id=grid_id, topology_id=topology_id, line_id=line_id, orders=ords)
    return sample_power_harmonics(grid_id,topology_id,line_id,ords)

@router.get("/measurements/{grid_id}/{topology_id}/power/line/{line_id}/thd", response_class=CustomJSONResponse)
async def get_power_thd(grid_id:int, topology_id:str,line_id:int,orders:str="1,3,5"):
    verify_line(grid_id,topology_id,line_id)
    ords=[int(o) for o in orders.split(",")]
    return sample_power_thd(grid_id,topology_id,line_id,ords)

@router.get("/measurements/{grid_id}/{topology_id}/power/lines", response_class=CustomJSONResponse)
async def get_all_line_powers(grid_id:int, topology_id:str):
    preview=verify_topology(grid_id,topology_id)
    results=[]
    for edge in preview.get("edges",[]):
        results.append(sample_power(grid_id,topology_id,edge["id"]))
    return results

# Injections
@router.get("/measurements/{grid_id}/{topology_id}/power/injections", response_class=CustomJSONResponse)
async def get_all_injection_powers(grid_id:int, topology_id:str):
    preview=verify_topology(grid_id,topology_id)
    results=[]
    for node in preview.get("nodes",[]):
        node_id=node["id"]
        results.append(sample_injection_power(grid_id,topology_id,node_id))
    return results

@router.get("/measurements/{grid_id}/{topology_id}/current/injections", response_class=CustomJSONResponse)
async def get_all_injection_currents(grid_id:int, topology_id:str):
    preview=verify_topology(grid_id,topology_id)
    results=[]
    for node in preview.get("nodes",[]):
        node_id=node["id"]
        results.append(sample_injection_current(grid_id,topology_id,node_id))
    return results

@router.get("/measurements/{grid_id}/{topology_id}/all", response_class=CustomJSONResponse)
async def get_all(grid_id: int, topology_id: str,
                  orders: str = Query("1,3,5,7")):
    preview = verify_topology(grid_id, topology_id)
    ords = [int(o) for o in orders.split(",")]
    nodes = []
    for n in preview.get("nodes", []):
        nid = n["id"]
        v = sample_bus_voltage(grid_id, topology_id, nid)["voltage"]
        h = sample_bus_harmonics(grid_id, topology_id, nid, ords)["harmonics"]
        nodes.append({"node_id": nid, "voltage": v, "harmonics": h})
    edges = []
    for e in preview.get("edges", []):
        lid = e["id"]
        i_val = sample_line_current(grid_id, topology_id, lid)["current"]
        pw = sample_power(grid_id, topology_id, lid)
        edges.append({"line_id": lid, "current": i_val, **pw})
    return {"nodes": nodes, "edges": edges}
app.include_router(router)

if __name__ == '__main__':
    uvicorn.run("mock_server:app", host=HOST, port=PORT)
