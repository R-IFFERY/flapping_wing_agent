"""
ZeroMQ client for online LBM/CFD coupling.
"""
import math

import zmq


class LBMClient:
    """REQ-side client for the LBM solver REP protocol."""

    def __init__(self, host="127.0.0.1", port=5555, wing_count=2, timeout_ms=600000):
        self.wing_count = int(wing_count)
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        self.sock.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
        self.sock.connect(f"tcp://{host}:{int(port)}")

    def close(self):
        self.sock.close(linger=0)

    def _send_recv(self, msg):
        self.sock.send_string(msg)
        return self.sock.recv_string()

    def reset(self):
        return self.parse_reply(self._send_recv("RESET 0"))

    def step(self, target_time, control_values):
        self._validate_target_time(target_time)
        expected_len = self.wing_count * 3
        if len(control_values) != expected_len:
            raise ValueError(f"control_values length must be {expected_len}")

        fields = [f"{target_time:.10f}"] + [f"{float(x):.10f}" for x in control_values]
        return self.parse_reply(self._send_recv("STEP " + " ".join(fields)))

    def save(self):
        return self.parse_reply(self._send_recv("SAVE"))

    def stop(self):
        return self.parse_reply(self._send_recv("STOP"))

    def parse_reply(self, reply):
        parts = reply.strip().split()
        if not parts:
            raise RuntimeError("empty reply")

        kind = parts[0]
        if kind == "ERROR":
            return {"type": "ERROR", "message": reply[len("ERROR"):].strip(), "raw_reply": reply}
        if kind == "STATUS":
            return {"type": "STATUS", "message": reply[len("STATUS"):].strip(), "raw_reply": reply}
        if kind == "FORCE":
            if len(parts) < 2:
                raise RuntimeError(f"invalid FORCE reply: {reply}")
            n = int(parts[1])
            values = [float(x) for x in parts[2:2 + n]]
            if len(values) != n:
                raise RuntimeError(f"FORCE length mismatch: header={n}, payload={len(values)}")
            parsed = self.parse_force_values(values)
            parsed["raw_reply"] = reply
            return parsed

        raise RuntimeError(f"unknown reply: {reply}")

    def parse_force_values(self, values):
        wc = self.wing_count
        base_force_len = 1 + 3 * wc + 3
        motion_len = 9 * wc
        expected_len = base_force_len + motion_len

        if len(values) != expected_len:
            raise RuntimeError(f"unexpected FORCE length: {len(values)}, expected {expected_len}")

        time = values[0]
        wing_force = []
        idx = 1
        for wing_idx in range(wc):
            wing_force.append({
                "wing": wing_idx + 1,
                "fx_avg": values[idx],
                "fy_avg": values[idx + 1],
                "m_avg": values[idx + 2],
            })
            idx += 3

        total = {
            "fx_avg": values[idx],
            "fy_avg": values[idx + 1],
            "m_avg": values[idx + 2],
        }
        idx += 3

        motion = []
        for wing_idx in range(wc):
            motion.append({
                "wing": wing_idx + 1,
                "sa": values[idx],
                "sau": values[idx + 1],
                "saac": values[idx + 2],
                "sb": values[idx + 3],
                "sbv": values[idx + 4],
                "sbac": values[idx + 5],
                "angle": values[idx + 6],
                "omega": values[idx + 7],
                "omgac": values[idx + 8],
            })
            idx += 9

        return {
            "type": "FORCE",
            "time": time,
            "wing_force": wing_force,
            "total_force": total,
            "motion_last": motion,
            "raw": values,
        }

    @staticmethod
    def _validate_target_time(target_time):
        scaled = float(target_time) / 0.0001
        if not math.isclose(scaled, round(scaled), rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("target_time must be an integer multiple of 0.0001")
