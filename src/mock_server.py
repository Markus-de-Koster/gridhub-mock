import json
import random
import math
from pathlib import Path
from datetime import datetime, timedelta
from fastapi import FastAPI, APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
# Base directory for mock data
DATA_DIR = Path(__file__).parent / "data"

class CustomJSONResponse(JSONResponse):
    def render(self, content: any) -> bytes:
        return json.dumps(content, allow_nan=True).encode("utf-8")

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

# ==== Random-walk helper ==== #

def random_walk(prev: float, sigma: float, bounds: tuple[float, float]) -> float:
    delta = random.gauss(0, sigma)
    new = prev + delta
    low, high = bounds
    return max(low, min(high, new))

# ==== Global state ==== #
_last_voltage = {}
_last_current = {}
_last_power = {}
_last_injection_current = {}
_last_injection_power = {}

# ==== Sampling functions ==== #

def sample_bus_voltage(bus_id: int):
    prev = _last_voltage.get(bus_id, 230.0)
    val = random_walk(prev, sigma=0.5, bounds=(210, 250))
    _last_voltage[bus_id] = val
    return {"bus_id": bus_id, "voltage": round(val, 2)}


def sample_bus_harmonics(bus_id: int, orders: list[int]):
    fund = _last_voltage.get(bus_id, 230.0)
    harms = {}
    for n in orders:
        base = fund * (0.01 / n)
        harms[n] = round(base * random.uniform(0.8, 1.2), 3)
    return {"bus_id": bus_id, "harmonics": harms}


def compute_thd(fund: float, harms: dict[int, float]) -> float:
    sum_sq = sum(v ** 2 for v in harms.values())
    return round(math.sqrt(sum_sq) / fund, 4)


def sample_bus_thd(bus_id: int, orders: list[int]):
    fund = _last_voltage.get(bus_id, 230.0)
    harms = sample_bus_harmonics(bus_id, orders)["harmonics"]
    return {"bus_id": bus_id, "thd": compute_thd(fund, harms)}


def sample_line_current(line_id: int):
    prev = _last_current.get(line_id, 10.0)
    val = random_walk(prev, sigma=0.2, bounds=(0, 50))
    _last_current[line_id] = val
    return {"line_id": line_id, "current": round(val, 3)}


def sample_current_harmonics(line_id: int, orders: list[int]):
    fund = _last_current.get(line_id, 10.0)
    harms = {}
    for n in orders:
        base = fund * (0.02 / n)
        harms[n] = round(base * random.uniform(0.8, 1.2), 4)
    return {"line_id": line_id, "harmonics": harms}


def sample_current_thd(line_id: int, orders: list[int]):
    fund = _last_current.get(line_id, 10.0)
    harms = sample_current_harmonics(line_id, orders)["harmonics"]
    return {"line_id": line_id, "thd": compute_thd(fund, harms)}


def sample_line_power(line_id: int):
    prev = _last_power.get(line_id, 500.0)
    val = random_walk(prev, sigma=50, bounds=(-5000, 5000))
    _last_power[line_id] = val
    return {"line_id": line_id, "power": round(val, 1)}


def sample_power_harmonics(line_id: int, orders: list[int]):
    fund = abs(_last_power.get(line_id, 500.0))
    harms = {}
    for n in orders:
        base = fund * (0.005 / n)
        harms[n] = round(base * random.uniform(0.8, 1.2), 2)
    return {"line_id": line_id, "harmonics": harms}


def sample_power_thd(line_id: int, orders: list[int]):
    fund = abs(_last_power.get(line_id, 500.0))
    harms = sample_power_harmonics(line_id, orders)["harmonics"]
    return {"line_id": line_id, "thd": compute_thd(fund, harms)}


def sample_injection_current(bus_id: int):
    prev = _last_injection_current.get(bus_id, 5.0)
    val = random_walk(prev, sigma=0.5, bounds=(0, 30))
    _last_injection_current[bus_id] = val
    return {"bus_id": bus_id, "injection_current": round(val, 3)}


def sample_injection_power(bus_id: int):
    prev = _last_injection_power.get(bus_id, 100.0)
    val = random_walk(prev, sigma=20, bounds=(-3000, 3000))
    _last_injection_power[bus_id] = val
    return {"bus_id": bus_id, "injection_power": round(val, 1)}


def sample_sequence(generator_fn, start: datetime, end: datetime, interval: int, **kwargs):
    t = start
    seq = []
    while t <= end:
        entry = generator_fn(**kwargs)
        entry["timestamp"] = t.isoformat()
        seq.append(entry)
        t += timedelta(seconds=interval)
    return seq

# ==== Voltage Endpoints ==== #
@router.get("/measurements/voltage/bus/{bus_id}", response_class=CustomJSONResponse)
async def get_voltage(bus_id: int,
                      start: str | None = Query(None),
                      end: str | None = Query(None),
                      interval: int = Query(60)):
    if start and end:
        return sample_sequence(sample_bus_voltage, datetime.fromisoformat(start), datetime.fromisoformat(end), interval, bus_id=bus_id)
    return sample_bus_voltage(bus_id)

@router.get("/measurements/voltage/bus/{bus_id}/harmonics", response_class=CustomJSONResponse)
async def get_voltage_harmonics(bus_id: int,
                                 orders: str = Query("1,3,5,7"),
                                 start: str | None = Query(None),
                                 end: str | None = Query(None),
                                 interval: int = Query(60)):
    ord_list = [int(o) for o in orders.split(",")]
    if start and end:
        return sample_sequence(sample_bus_harmonics, datetime.fromisoformat(start), datetime.fromisoformat(end), interval, bus_id=bus_id, orders=ord_list)
    return sample_bus_harmonics(bus_id, ord_list)

@router.get("/measurements/voltage/bus/{bus_id}/thd", response_class=CustomJSONResponse)
async def get_voltage_thd(bus_id: int,
                          orders: str = Query("1,3,5,7")):
    ord_list = [int(o) for o in orders.split(",")]
    return sample_bus_thd(bus_id, ord_list)

@router.get("/measurements/voltage/buses", response_class=CustomJSONResponse)
async def get_all_bus_voltages():
    grids_file = DATA_DIR / "grids.json"
    if not grids_file.exists():
        raise HTTPException(status_code=500, detail="grids.json missing")
    grids = json.loads(grids_file.read_text())
    results = []
    for grid in grids:
        for bus_id in grid.get("structure", {}).get("nodes", []):
            results.append(sample_bus_voltage(bus_id))
    return results

# ==== Current Endpoints ==== #
@router.get("/measurements/current/line/{line_id}", response_class=CustomJSONResponse)
async def get_line_current(line_id: int,
                           start: str | None = Query(None),
                           end: str | None = Query(None),
                           interval: int = Query(60)):
    if start and end:
        return sample_sequence(sample_line_current, datetime.fromisoformat(start), datetime.fromisoformat(end), interval, line_id=line_id)
    return sample_line_current(line_id)

@router.get("/measurements/current/line/{line_id}/harmonics", response_class=CustomJSONResponse)
async def get_current_harmonics(line_id: int,
                                orders: str = Query("1,3,5"),
                                start: str | None = Query(None),
                                end: str | None = Query(None),
                                interval: int = Query(60)):
    ord_list = [int(o) for o in orders.split(",")]
    if start and end:
        return sample_sequence(sample_current_harmonics, datetime.fromisoformat(start), datetime.fromisoformat(end), interval, line_id=line_id, orders=ord_list)
    return sample_current_harmonics(line_id, ord_list)

@router.get("/measurements/current/line/{line_id}/thd", response_class=CustomJSONResponse)
async def get_current_thd(line_id: int,
                           orders: str = Query("1,3,5")):
    ord_list = [int(o) for o in orders.split(",")]
    return sample_current_thd(line_id, ord_list)

@router.get("/measurements/current/lines", response_class=CustomJSONResponse)
async def get_all_line_currents():
    # Assume line IDs from 1 to N; replace N as needed or parse from data
    return [sample_line_current(i) for i in range(1, 11)]

# ==== Power Endpoints ==== #
@router.get("/measurements/power/line/{line_id}", response_class=CustomJSONResponse)
async def get_line_power(line_id: int,
                         start: str | None = Query(None),
                         end: str | None = Query(None),
                         interval: int = Query(60)):
    if start and end:
        return sample_sequence(sample_line_power, datetime.fromisoformat(start), datetime.fromisoformat(end), interval, line_id=line_id)
    return sample_line_power(line_id)

@router.get("/measurements/power/line/{line_id}/harmonics", response_class=CustomJSONResponse)
async def get_power_harmonics(line_id: int,
                               orders: str = Query("1,3,5"),
                               start: str | None = Query(None),
                               end: str | None = Query(None),
                               interval: int = Query(60)):
    ord_list = [int(o) for o in orders.split(",")]
    if start and end:
        return sample_sequence(sample_power_harmonics, datetime.fromisoformat(start), datetime.fromisoformat(end), interval, line_id=line_id, orders=ord_list)
    return sample_power_harmonics(line_id, ord_list)

@router.get("/measurements/power/line/{line_id}/thd", response_class=CustomJSONResponse)
async def get_power_thd(line_id: int,
                          orders: str = Query("1,3,5")):
    ord_list = [int(o) for o in orders.split(",")]
    return sample_power_thd(line_id, ord_list)

@router.get("/measurements/power/lines", response_class=CustomJSONResponse)
async def get_all_line_powers():
    return [sample_line_power(i) for i in range(1, 11)]

# ==== Injection Endpoints ==== #
@router.get("/measurements/power/injections", response_class=CustomJSONResponse)
async def get_all_injection_powers():
    grids = json.loads((DATA_DIR / "grids.json").read_text())
    injections = []
    for grid in grids:
        for bus_id in grid.get("structure", {}).get("nodes", []):
            injections.append(sample_injection_power(bus_id))
    return injections

@router.get("/measurements/current/injections", response_class=CustomJSONResponse)
async def get_all_injection_currents():
    grids = json.loads((DATA_DIR / "grids.json").read_text())
    injections = []
    for grid in grids:
        for bus_id in grid.get("structure", {}).get("nodes", []):
            injections.append(sample_injection_current(bus_id))
    return injections

app.include_router(router)
