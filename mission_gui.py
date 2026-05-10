#!/usr/bin/env python3
"""Autonomous mission GUI for service mission execution."""

from __future__ import annotations

import argparse
import json
import math
import queue
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node

import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText


STATUS_TEXT = {
    GoalStatus.STATUS_UNKNOWN: "UNKNOWN",
    GoalStatus.STATUS_ACCEPTED: "ACCEPTED",
    GoalStatus.STATUS_EXECUTING: "EXECUTING",
    GoalStatus.STATUS_CANCELING: "CANCELING",
    GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
    GoalStatus.STATUS_CANCELED: "CANCELED",
    GoalStatus.STATUS_ABORTED: "ABORTED",
}

NAV2_ACTION_TYPE = "nav2_msgs/action/NavigateToPose"

LOCATION_DISPLAY = {
    "dock": "dock",
    "supermarket": "supermarket",
    "restaurant": "restaurant",
    "fire_center": "fire_center",
    "pharmacy": "pharmacy",
    "house_1": "house_1",
    "house_2": "house_2",
    "house_3": "house_3",
    "house_4": "house_4",
    "house_5": "house_5",
}

MISSION_DEFINITIONS = {
    "Grocery delivery": "supermarket",
    "Food delivery": "restaurant",
    "Fire emergency": "fire_center",
    "Medical help": "pharmacy",
}

LOCATION_ALIASES = {
    "dock": "docking_station",
    "docking": "docking_station",
    "fire_center": "fire_center",
    "firefightingcentre": "fire_center",
    "firefighting_center": "fire_center",
    "firefighting_station": "fire_center",
    "fire_station": "fire_center",
    "firestation": "fire_center",
}


class MissionError(RuntimeError):
    """Raised for mission planning or execution failures."""


@dataclass
class LandmarkPose:
    frame: str
    x: float
    y: float
    z: float
    yaw: float


def normalize_location_name(raw_name: str) -> str:
    """Normalize user and QR landmark names to canonical keys."""
    name = raw_name.strip().lower()
    if ":" in name:
        _, name = name.split(":", 1)
    name = name.replace("-", "_").replace(" ", "_")
    name = re.sub(r"_+", "_", name).strip("_")

    house_match = re.fullmatch(r"house_?(\d+)", name)
    if house_match:
        return f"house_{int(house_match.group(1))}"

    return LOCATION_ALIASES.get(name, name)


def display_name(location_name: str) -> str:
    canonical = normalize_location_name(location_name)
    return LOCATION_DISPLAY.get(canonical, canonical.replace("_", " ").title())


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    half = yaw * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


class MissionController(Node):
    """ROS2 mission controller that sends Nav2 goals from landmark coordinates."""

    def __init__(
        self,
        landmarks_path: Path,
        action_name: str = "/navigate_to_pose",
        zero_goal_stamp: bool = True,
    ):
        super().__init__("mission_controller_gui")
        self.landmarks_path = landmarks_path
        self.requested_action_name = action_name
        self.zero_goal_stamp = zero_goal_stamp
        self.nav_clients: dict[str, ActionClient] = {}
        self.active_action_name: str | None = None

        self.goal_pose_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self._ensure_nav_client(self.requested_action_name)
        self.landmarks: dict[str, LandmarkPose] = {}
        self.feedback_every_n = 10
        self.reload_landmarks()

    def reload_landmarks(self) -> None:
        if not self.landmarks_path.exists():
            raise MissionError(f"Landmarks file not found: {self.landmarks_path}")

        with self.landmarks_path.open("r", encoding="utf-8") as file:
            raw_data = json.load(file)

        landmarks: dict[str, LandmarkPose] = {}
        for raw_name, pose in raw_data.items():
            canonical = normalize_location_name(raw_name)
            try:
                landmarks[canonical] = LandmarkPose(
                    frame=pose.get("frame", "map"),
                    x=float(pose["x"]),
                    y=float(pose["y"]),
                    z=float(pose.get("z", 0.0)),
                    yaw=float(pose.get("yaw", 0.0)),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise MissionError(
                    f"Invalid landmark format for '{raw_name}' in {self.landmarks_path}"
                ) from exc

        self.landmarks = landmarks

    def build_mission_route(self, mission_name: str, house_name: str) -> list[str]:
        service_point = MISSION_DEFINITIONS.get(mission_name)
        if service_point is None:
            supported = ", ".join(MISSION_DEFINITIONS.keys())
            raise MissionError(f"Unknown mission '{mission_name}'. Supported: {supported}")

        house = normalize_location_name(house_name)
        if not re.fullmatch(r"house_[1-5]", house):
            raise MissionError("House must be one of: house_1, house_2, house_3, house_4, house_5")

        return ["docking_station", service_point, house, "docking_station"]

    def execute_mission(
        self,
        mission_name: str,
        house_name: str,
        status_callback: Callable[[str], None],
    ) -> None:
        self.reload_landmarks()
        requested_route = self.build_mission_route(mission_name, house_name)

        missing = []
        executable_route = []
        for location in requested_route:
            if location in self.landmarks:
                executable_route.append(location)
            elif location not in missing:
                missing.append(location)

        if missing:
            missing_display = ", ".join(display_name(name) for name in missing)
            status_callback(f"[WARN] Missing landmarks skipped: {missing_display}")

        if not executable_route:
            available_display = ", ".join(
                display_name(name) for name in sorted(self.landmarks.keys())
            )
            raise MissionError(
                "No mission waypoint can be executed from current landmarks. "
                f"Available landmarks: {available_display}"
            )

        status_callback(f"Mission started: {mission_name} to {display_name(house_name)}")
        for index, location in enumerate(executable_route, start=1):
            status_callback(f"[{index}/{len(executable_route)}] Navigating to {display_name(location)}")
            self.navigate_to(location, status_callback)

        if "docking_station" in executable_route:
            status_callback("Mission complete.")
        else:
            status_callback("[WARN] Mission complete without docking_station.")

    def navigate_to(self, location_name: str, status_callback: Callable[[str], None]) -> None:
        canonical = normalize_location_name(location_name)
        target = self.landmarks[canonical]
        goal = NavigateToPose.Goal()
        goal.pose = self._build_pose_stamped(target)

        # Publish the same goal on /goal_pose for compatibility and easy debugging.
        self.goal_pose_pub.publish(goal.pose)
        status_callback(
            "Published /goal_pose: "
            f"{display_name(canonical)} "
            f"(x={target.x:.2f}, y={target.y:.2f}, yaw={target.yaw:.2f})"
        )

        nav_client, nav_action_name = self._select_nav_action_client(status_callback)
        status_callback(f"Sending Nav2 action goal to {nav_action_name}")

        send_future = nav_client.send_goal_async(
            goal,
            feedback_callback=self._make_feedback_callback(canonical, status_callback),
        )
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()

        if goal_handle is None or not goal_handle.accepted:
            raise MissionError(f"Goal rejected for {display_name(canonical)}.")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result_wrapper = result_future.result()
        if result_wrapper is None:
            raise MissionError(f"No result returned for {display_name(canonical)}.")

        status = result_wrapper.status
        if status != GoalStatus.STATUS_SUCCEEDED:
            text = STATUS_TEXT.get(status, str(status))
            details = self._extract_nav2_result_details(result_wrapper)
            raise MissionError(
                f"Navigation to {display_name(canonical)} failed: {text}.{details}"
            )

        status_callback(f"Reached {display_name(canonical)}")

    def _duration_to_seconds(self, duration_msg: object) -> float | None:
        if duration_msg is None:
            return None
        sec = getattr(duration_msg, "sec", None)
        nanosec = getattr(duration_msg, "nanosec", None)
        if sec is None or nanosec is None:
            return None
        return float(sec) + float(nanosec) / 1_000_000_000.0

    def _make_feedback_callback(
        self,
        canonical_target: str,
        status_callback: Callable[[str], None],
    ) -> Callable[[object], None]:
        counter = {"n": 0}

        def _callback(feedback_msg: object) -> None:
            counter["n"] += 1
            if counter["n"] % self.feedback_every_n != 0:
                return

            feedback = getattr(feedback_msg, "feedback", None)
            if feedback is None:
                status_callback(f"[FB] {display_name(canonical_target)}: feedback received")
                return

            distance_remaining = getattr(feedback, "distance_remaining", None)
            recoveries = getattr(feedback, "number_of_recoveries", None)
            eta = self._duration_to_seconds(getattr(feedback, "estimated_time_remaining", None))
            nav_time = self._duration_to_seconds(getattr(feedback, "navigation_time", None))

            parts = [f"[FB] {display_name(canonical_target)}"]
            if distance_remaining is not None:
                parts.append(f"dist={float(distance_remaining):.2f}m")
            if eta is not None:
                parts.append(f"eta={eta:.1f}s")
            if nav_time is not None:
                parts.append(f"nav_time={nav_time:.1f}s")
            if recoveries is not None:
                parts.append(f"recoveries={int(recoveries)}")

            status_callback(" | ".join(parts))

        return _callback

    def _ensure_nav_client(self, action_name: str) -> ActionClient:
        if action_name not in self.nav_clients:
            self.nav_clients[action_name] = ActionClient(self, NavigateToPose, action_name)
        return self.nav_clients[action_name]

    def _discover_nav_action_names(self) -> list[str]:
        if not hasattr(self, "get_action_names_and_types"):
            return []

        names: list[str] = []
        try:
            for action_name, action_types in self.get_action_names_and_types():
                if NAV2_ACTION_TYPE in action_types:
                    names.append(action_name)
        except Exception:
            # Keep mission execution compatible with ROS 2 distros that do not
            # support action graph introspection on Node.
            return []
        return names

    def _select_nav_action_client(
        self, status_callback: Callable[[str], None]
    ) -> tuple[ActionClient, str]:
        candidates = [self.requested_action_name, "/navigate_to_pose", "navigate_to_pose"]
        candidates.extend(self._discover_nav_action_names())

        deduped: list[str] = []
        for name in candidates:
            if name and name not in deduped:
                deduped.append(name)

        for action_name in deduped:
            client = self._ensure_nav_client(action_name)
            if client.wait_for_server(timeout_sec=1.0):
                if self.active_action_name != action_name:
                    self.active_action_name = action_name
                    status_callback(f"Using Nav2 action server: {action_name}")
                return client, action_name

        discovered = self._discover_nav_action_names()
        discovered_display = ", ".join(discovered) if discovered else "none"
        checked_display = ", ".join(deduped)
        raise MissionError(
            "Nav2 action server is not reachable. "
            f"Checked: {checked_display}. Discovered NavigateToPose servers: {discovered_display}."
        )

    def _extract_nav2_result_details(self, result_wrapper: object) -> str:
        # Newer Nav2 releases provide error_code + error_msg in result payload.
        result_payload = getattr(result_wrapper, "result", None)
        if result_payload is None:
            return ""

        error_code = getattr(result_payload, "error_code", None)
        error_msg = getattr(result_payload, "error_msg", "")
        if error_code is None and not error_msg:
            return ""

        detail = ""
        if error_code is not None:
            detail += f" error_code={error_code}"
        if error_msg:
            detail += f" error_msg={error_msg}"
        return detail

    def _build_pose_stamped(self, pose: LandmarkPose) -> PoseStamped:
        msg = PoseStamped()
        msg.header.frame_id = pose.frame
        if not self.zero_goal_stamp:
            msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = pose.x
        msg.pose.position.y = pose.y
        msg.pose.position.z = pose.z

        qx, qy, qz, qw = yaw_to_quaternion(pose.yaw)
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        return msg


class MissionGUI:
    """Tkinter GUI for selecting and running autonomous missions."""

    def __init__(self, root: tk.Tk, controller: MissionController):
        self.root = root
        self.controller = controller
        self.worker_thread: threading.Thread | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.is_closed = False

        self.root.title("Autonomous Mission Execution")
        self.root.geometry("760x500")
        self.root.minsize(700, 460)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.mission_var = tk.StringVar(value=list(MISSION_DEFINITIONS.keys())[0])
        self.house_var = tk.StringVar(value="house_1")
        self.route_var = tk.StringVar(value="")
        self.use_custom_var = tk.BooleanVar(value=False)
        self.custom_mission_name_var = tk.StringVar(value="")
        self.custom_destination_var = tk.StringVar(value="")

        self._build_layout()
        self._refresh_route_preview()
        self._start_log_pump()
        self.load_landmarks(initial=True)

    def _build_layout(self) -> None:
        main = ttk.Frame(self.root, padding=16)
        main.pack(fill="both", expand=True)

        title = ttk.Label(
            main,
            text="Project ",
            font=("TkDefaultFont", 13, "bold"),
        )
        title.pack(anchor="w")

        subtitle = ttk.Label(
            main,
            text="Select mission type and target house. The robot handles the rest autonomously.",
        )
        subtitle.pack(anchor="w", pady=(4, 14))

        form = ttk.Frame(main)
        form.pack(fill="x")

        ttk.Label(form, text="Mission type:").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=6)
        self.mission_box = ttk.Combobox(
            form,
            textvariable=self.mission_var,
            values=list(MISSION_DEFINITIONS.keys()),
            state="readonly",
            width=24,
        )
        self.mission_box.grid(row=0, column=1, sticky="w", pady=6)
        self.mission_box.bind("<<ComboboxSelected>>", lambda _: self._refresh_route_preview())

        ttk.Label(form, text="Target house:").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=6)
        house_values = [f"house_{i}" for i in range(1, 6)]
        self.house_box = ttk.Combobox(
            form,
            textvariable=self.house_var,
            values=house_values,
            state="readonly",
            width=24,
        )
        self.house_box.grid(row=1, column=1, sticky="w", pady=6)
        self.house_box.bind("<<ComboboxSelected>>", lambda _: self._refresh_route_preview())

        # Custom mission section
        separator = ttk.Separator(main, orient="horizontal")
        separator.pack(fill="x", pady=(12, 12))

        custom_label = ttk.Label(
            main,
            text="Custom Mission",
            font=("TkDefaultFont", 10, "bold"),
        )
        custom_label.pack(anchor="w")

        custom_frame = ttk.Frame(main)
        custom_frame.pack(fill="x", pady=(6, 10))

        self.custom_check = ttk.Checkbutton(
            custom_frame,
            text="Use Custom Mission",
            variable=self.use_custom_var,
            command=self._on_custom_mission_toggled,
        )
        self.custom_check.pack(anchor="w")

        custom_form = ttk.Frame(main)
        custom_form.pack(fill="x")

        ttk.Label(custom_form, text="Mission name:").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=6)
        self.custom_mission_entry = ttk.Entry(
            custom_form,
            textvariable=self.custom_mission_name_var,
            width=24,
            state="disabled",
        )
        self.custom_mission_entry.grid(row=0, column=1, sticky="w", pady=6)
        self.custom_mission_entry.bind("<KeyRelease>", lambda _: self._refresh_route_preview())

        ttk.Label(custom_form, text="Destination(s):").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=6)
        self.custom_destination_entry = ttk.Entry(
            custom_form,
            textvariable=self.custom_destination_var,
            width=24,
            state="disabled",
        )
        self.custom_destination_entry.grid(row=1, column=1, sticky="w", pady=6)
        self.custom_destination_entry.bind("<KeyRelease>", lambda _: self._refresh_route_preview())

        hint = ttk.Label(
            main,
            text="(Enter comma-separated location names for multiple destinations, e.g., 'supermarket, pharmacy, restaurant')",
            font=("TkDefaultFont", 8),
        )
        hint.pack(anchor="w", pady=(0, 10))

        controls = ttk.Frame(main)
        controls.pack(fill="x", pady=(10, 8))

        self.start_btn = ttk.Button(controls, text="Start Mission", command=self.start_mission)
        self.start_btn.pack(side="left")

        self.reload_btn = ttk.Button(controls, text="Reload Landmarks", command=self.load_landmarks)
        self.reload_btn.pack(side="left", padx=(8, 0))

        self.route_label = ttk.Label(main, textvariable=self.route_var)
        self.route_label.pack(anchor="w", pady=(8, 10))

        self.log_text = ScrolledText(main, height=16, wrap="word", state="disabled")
        self.log_text.pack(fill="both", expand=True)

    def _on_custom_mission_toggled(self) -> None:
        state = "normal" if self.use_custom_var.get() else "disabled"
        self.custom_mission_entry.configure(state=state)
        self.custom_destination_entry.configure(state=state)
        self.mission_box.configure(state="disabled" if self.use_custom_var.get() else "readonly")
        self.house_box.configure(state="disabled" if self.use_custom_var.get() else "readonly")
        self._refresh_route_preview()

    def _refresh_route_preview(self) -> None:
        if self.use_custom_var.get():
            mission_name = self.custom_mission_name_var.get().strip()
            destinations_str = self.custom_destination_var.get().strip()
            if mission_name and destinations_str:
                destinations = [d.strip() for d in destinations_str.split(",") if d.strip()]
                if destinations:
                    route = ["docking_station"] + [normalize_location_name(d) for d in destinations] + ["docking_station"]
                    rendered = " -> ".join(display_name(point) for point in route)
                    self.route_var.set(f"Route: {rendered}")
                else:
                    self.route_var.set("Route: (enter at least one destination)")
            else:
                self.route_var.set("Route: (enter mission name and destination(s))")
        else:
            mission = self.mission_var.get()
            house = normalize_location_name(self.house_var.get())
            route = self.controller.build_mission_route(mission, house)
            rendered = " -> ".join(display_name(point) for point in route)
            self.route_var.set(f"Route: {rendered}")

    def _start_log_pump(self) -> None:
        self.root.after(100, self._pump_log_queue)

    def _pump_log_queue(self) -> None:
        if self.is_closed:
            return
        while not self.log_queue.empty():
            line = self.log_queue.get_nowait()
            self.log_text.configure(state="normal")
            self.log_text.insert("end", f"{line}\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
        self.root.after(100, self._pump_log_queue)

    def _safe_after(self, callback: Callable[[], None]) -> None:
        if self.is_closed:
            return
        try:
            self.root.after(0, callback)
        except tk.TclError:
            self.is_closed = True

    def log(self, message: str) -> None:
        self.log_queue.put(message)

    def load_landmarks(self, initial: bool = False) -> None:
        try:
            self.controller.reload_landmarks()
        except MissionError as exc:
            self.log(f"[ERROR] {exc}")
            if not initial:
                messagebox.showerror("Landmarks Error", str(exc))
            return

        count = len(self.controller.landmarks)
        file_path = str(self.controller.landmarks_path)
        self.log(f"Loaded {count} landmarks from {file_path}")

    def start_mission(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showwarning("Mission Busy", "A mission is already running.")
            return

        if self.use_custom_var.get():
            mission_name = self.custom_mission_name_var.get().strip()
            destinations_str = self.custom_destination_var.get().strip()
            if not mission_name:
                messagebox.showerror("Invalid Mission", "Please enter a mission name.")
                return
            if not destinations_str:
                messagebox.showerror("Invalid Mission", "Please enter at least one destination.")
                return
            destinations = [d.strip() for d in destinations_str.split(",") if d.strip()]
            if not destinations:
                messagebox.showerror("Invalid Mission", "Please enter at least one valid destination.")
                return
        else:
            mission_name = self.mission_var.get()
            destination = normalize_location_name(self.house_var.get())
            destinations = [destination]

        self._refresh_route_preview()
        self.start_btn.configure(state="disabled")
        self.reload_btn.configure(state="disabled")

        def worker() -> None:
            try:
                if self.use_custom_var.get():
                    self._execute_custom_mission(mission_name, destinations, self.log)
                else:
                    self.controller.execute_mission(mission_name, destination, self.log)
            except MissionError as exc:
                self.log(f"[ERROR] {exc}")
                self._safe_after(lambda: messagebox.showerror("Mission Failed", str(exc)))
            except Exception as exc:  # pragma: no cover
                self.log(f"[ERROR] Unexpected failure: {exc}")
                self._safe_after(lambda: messagebox.showerror("Mission Failed", str(exc)))
            finally:
                self._safe_after(self._finish_mission_ui)

        if self.use_custom_var.get():
            destinations_display = ", ".join(display_name(d) for d in destinations)
            self.log(f"Queued custom mission: {mission_name} to {destinations_display}")
        else:
            self.log(f"Queued mission: {mission_name} to {display_name(destination)}")
        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def _execute_custom_mission(self, mission_name: str, destinations: list[str], status_callback: Callable[[str], None]) -> None:
        """Execute a custom mission: dock -> destinations -> dock."""
        normalized_dests = [normalize_location_name(d) for d in destinations]
        route = ["docking_station"] + normalized_dests + ["docking_station"]
        
        missing = []
        executable_route = []
        for location in route:
            if location in self.controller.landmarks:
                executable_route.append(location)
            elif location not in missing:
                missing.append(location)

        if missing:
            missing_display = ", ".join(display_name(name) for name in missing)
            status_callback(f"[WARN] Missing landmarks skipped: {missing_display}")

        if not executable_route:
            available_display = ", ".join(
                display_name(name) for name in sorted(self.controller.landmarks.keys())
            )
            raise MissionError(
                "No custom mission waypoint can be executed from current landmarks. "
                f"Available landmarks: {available_display}"
            )

        destinations_display = ", ".join(display_name(d) for d in normalized_dests)
        status_callback(f"Custom mission started: {mission_name} to {destinations_display}")
        for index, location in enumerate(executable_route, start=1):
            status_callback(f"[{index}/{len(executable_route)}] Navigating to {display_name(location)}")
            self.controller.navigate_to(location, status_callback)

        if "docking_station" in executable_route:
            status_callback("Custom mission complete.")
        else:
            status_callback("[WARN] Custom mission complete without docking_station.")

    def _finish_mission_ui(self) -> None:
        self.start_btn.configure(state="normal")
        self.reload_btn.configure(state="normal")
        if self.use_custom_var.get():
            self.custom_mission_entry.focus()

    def on_close(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            if not messagebox.askyesno(
                "Mission Running",
                "A mission is still running. Close GUI anyway?",
            ):
                return
        self.is_closed = True
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    def parse_bool(value: str) -> bool:
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

    parser = argparse.ArgumentParser(description="Autonomous mission execution GUI")
    parser.add_argument(
        "--landmarks",
        type=Path,
        default=Path(__file__).resolve().parent / "qr_landmarks.json",
        help="Path to JSON landmarks file (default: ./qr_landmarks.json)",
    )
    parser.add_argument(
        "--action-name",
        type=str,
        default="/navigate_to_pose",
        help="Nav2 NavigateToPose action name (default: /navigate_to_pose)",
    )
    parser.add_argument(
        "--zero-goal-stamp",
        type=parse_bool,
        default=True,
        help="Send goals with stamp=0 to avoid TF time mismatch (default: true)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()

    controller: MissionController | None = None
    try:
        controller = MissionController(
            args.landmarks,
            action_name=args.action_name,
            zero_goal_stamp=args.zero_goal_stamp,
        )
        root = tk.Tk()
        app = MissionGUI(root, controller)
        app.log("GUI ready. Select mission and house, then click Start Mission.")
        root.mainloop()
    finally:
        if controller is not None:
            controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
