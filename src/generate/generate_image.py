import logging
import math
import random
from enum import Enum
import numpy as np
from matplotlib import pyplot as plt, patches
from matplotlib.transforms import Affine2D
from waymo_open_dataset.protos.map_pb2 import TrafficSignalLaneState, LaneCenter, LaneNeighbor
from waymo_open_dataset.protos.scenario_pb2 import Scenario, Track, ObjectState

from config import DPI, ANNOTATE_TEXT_SIZE, COLOR_AUTONOMOUS_VEHICLE, COLOR_PEDESTRIAN_CYCLIST, \
    TRAFFIC_SIGNAL_COLOR_STATE_UNKNOWN, TRAFFIC_SIGNAL_COLOR_STATE_GO, TRAFFIC_SIGNAL_COLOR_STATE_CAUTION, \
    TRAFFIC_SIGNAL_COLOR_STATE_STOP, COLOR_MAP_SEED
from src.utils import maths
from src.utils.geometry import ccw_sort
from src.utils.maths import get_agent_speed
from src.utils.misc import get_intersection_lane_changes_cache, get_agents_in_intersections_at_tick, \
    is_agent_in_lane_change_intersection_at_tick

logger = logging.getLogger(__name__)


def create_figure_and_axes(size_pixels, white=False):
    """Initializes a unique figure and axes for plotting."""
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Sets output image to pixel resolution.
    dpi = DPI
    size_inches = size_pixels / dpi
    fig.set_size_inches([size_inches, size_inches])
    fig.set_dpi(dpi)

    if white:
        fig.set_facecolor('white')
        ax.set_facecolor('white')
        ax.xaxis.label.set_color('black')
        ax.tick_params(axis='x', colors='black')
        ax.yaxis.label.set_color('black')
        ax.tick_params(axis='y', colors='black')
        ax.title.set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['left'].set_color('black')
    else:
        fig.set_facecolor('black')
        ax.set_facecolor('black')
        ax.xaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='y', colors='white')
        ax.title.set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')
    fig.set_tight_layout(True)
    ax.grid(False)
    return fig, ax


def fig_canvas_image(fig):
    """Returns a [H, W, 3] uint8 np.array image from fig.canvas.tostring_rgb()."""
    # Just enough margin in the figure to display xticks and yticks.
    fig.subplots_adjust(
        left=0.08, bottom=0.08, right=0.98, top=0.98, wspace=0.0, hspace=0.0)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))


def get_viewport(
        agent_tracks: list[Track]
):
    # [num_agents, num_states]
    track_states: list[list[ObjectState]] = [track.states for track in agent_tracks]

    # All x and y coordinates from agents throughout states
    all_y = []
    all_x = []

    for agent_track_states in track_states:
        # Add all agent state position where state is valid
        all_y.extend([state.center_y for state in agent_track_states if state.valid])
        all_x.extend([state.center_x for state in agent_track_states if state.valid])

    center_y = (np.max(all_y) + np.min(all_y)) / 2
    center_x = (np.max(all_x) + np.min(all_x)) / 2

    range_y = np.ptp(all_y)
    range_x = np.ptp(all_x)

    width = max(range_y, range_x)

    return center_y, center_x, width


class AnnotateType(Enum):
    WITH_ID = "id"
    WITH_ID_AND_SPEED = "id+speed"


def visualize_one_step_with_agents(step: int,
                                   scenario: Scenario,
                                   title,
                                   annotate_type: AnnotateType,
                                   center_y,
                                   center_x,
                                   width,
                                   agent_labels,
                                   kd_tree,
                                   size_pixels=1000,
                                   plot: bool = False,
                                   white=False):
    # Create figure and axes.
    fig, ax = create_figure_and_axes(size_pixels=size_pixels, white=white)

    # Visualize scene
    # Comment or uncomment to change the visualization
    visualize_intersection(scenario, ax)
    visualize_crosswalks(scenario, ax, white=white)
    visualize_lane_centers(scenario, ax, white=white)
    visualize_road_edges(scenario, ax, white=white)
    # visualize_road_center_neighbors_ids(scenario, ax)
    # visualize_road_type(scenario, ax)
    visualize_traffic_lights(scenario, ax, step)
    visualize_agents(scenario, ax, step, annotate_type, agent_labels, kd_tree, white=white)

    # Debug
    # visualize_road_center_ids(scenario, ax)
    # visualize_road_edge_ids(scenario, ax)

    ax.set_title(title)

    # Set axes.  Should be at least 10m on a side and cover 160% of agents.
    size = max(10, width * 1.0)
    ax.axis([
        -size / 2 + center_x, size / 2 + center_x, -size / 2 + center_y,
        size / 2 + center_y
    ])
    ax.set_aspect('equal')

    image = fig_canvas_image(fig)

    if plot:
        plt.show()

    plt.close(fig)

    return image


def visualize_all_agents_smooth_one_step(
        scenario: Scenario,
        annotate_type: AnnotateType,
        agent_labels,
        kd_tree,
        size_pixels=1000,
        step: int = 0,
        white=False,
):
    center_y, center_x, width = get_viewport(scenario.tracks)

    visualize_one_step_with_agents(
        step, scenario, str(step), annotate_type, center_y, center_x, width, agent_labels, kd_tree, size_pixels, True,
        white=white
    )


def visualize_all_agents_smooth(
        scenario: Scenario,
        annotate_type: AnnotateType,
        agent_labels,
        kd_tree,
        size_pixels=1000,
        white=False,

):
    center_y, center_x, width = get_viewport(scenario.tracks)

    images = []

    for step in range(91):
        time_seg = "past"

        if step == 10:
            time_seg = "current"
        elif step > 10:
            time_seg = "future"

        title = f"scenario: {scenario.scenario_id}, {time_seg}: {step}"

        images.append(
            visualize_one_step_with_agents(
                step, scenario, title, annotate_type, center_y, center_x, width, agent_labels, kd_tree, size_pixels,
                white=white
            )
        )

    return images


def visualize_lane_centers(scenario: Scenario, ax, white=False):
    road_center_y = []
    road_center_x = []

    road_center_features = [map_feature for map_feature in scenario.map_features
                            if map_feature.WhichOneof("feature_data") == "lane"]

    colors_map_road_center = get_colors_map([center_feature.id for center_feature in road_center_features], white=white)

    road_center_color = []
    for road_center_feature in road_center_features:
        feature: LaneCenter = road_center_feature.lane
        if feature.type == LaneCenter.LaneType.TYPE_BIKE_LANE:
            road_center_color.extend(["grey" for _ in feature.polyline])
        else:
            road_center_color.extend([colors_map_road_center[road_center_feature.id] for _ in feature.polyline])

        road_center_y.extend([polyline.y for polyline in feature.polyline])
        road_center_x.extend([polyline.x for polyline in feature.polyline])

    ax.scatter(road_center_x, road_center_y, marker='o', alpha=0.4, s=1, color=road_center_color)


def visualize_road_edges(scenario: Scenario, ax, white=False):
    road_line_edge_y = []
    road_line_edge_x = []
    road_line_edge_colors = []

    road_lane_types = ["road_line", "road_edge"]

    road_line_edge_features = [map_feature for map_feature in scenario.map_features
                               if map_feature.WhichOneof("feature_data") in road_lane_types]

    colors_map_road_line_edge = get_colors_map([edge_feature.id for edge_feature in road_line_edge_features],
                                               white=white)

    for road_line_edge_feature in road_line_edge_features:
        feature_data_type = road_line_edge_feature.WhichOneof("feature_data")
        feature = getattr(road_line_edge_feature, feature_data_type)
        road_line_edge_colors.extend(
            [colors_map_road_line_edge[road_line_edge_feature.id] for polyline in feature.polyline])
        road_line_edge_y.extend([polyline.y for polyline in feature.polyline])
        road_line_edge_x.extend([polyline.x for polyline in feature.polyline])

    if white:
        ax.scatter(road_line_edge_x, road_line_edge_y, marker='o', alpha=1, s=1.5, color="black")
    else:
        ax.scatter(road_line_edge_x, road_line_edge_y, marker='o', alpha=1, s=1.5, color="white")


def visualize_agents(scenario: Scenario, ax, step, annotate_type, agents_labels, kd_tree, white=False):
    masked_y = []
    masked_x = []
    colors = []
    color_map = get_colors_map([track.id for track in scenario.tracks], white=white)

    intersections, lane_changes = get_intersection_lane_changes_cache(scenario)
    agents_in_intersection = get_agents_in_intersections_at_tick(scenario, intersections, tick=step)

    for agent_track in scenario.tracks:

        # Add agent state position if state is valid
        if agent_track.states[step].valid:
            state = agent_track.states[step]

            masked_y.append(state.center_y)
            masked_x.append(state.center_x)
            colors.append(color_map[agent_track.id])

            # The heading of an agent is stored as radian normalized in [-pi, pi),
            # thus we convert it to degrees in [0, 360)
            angle = (math.degrees(state.heading) - 90) % 360

            rect = patches.Rectangle(
                (state.center_x - state.width / 2, state.center_y - state.length / 2),
                state.width,
                state.length,
                linewidth=1,
                facecolor=color_map[agent_track.id],
                transform=Affine2D().rotate_deg_around(*(state.center_x, state.center_y), angle) + ax.transData,
                alpha=0.4
            )

            ax.scatter([state.center_x], [state.center_y], marker='o', alpha=1, s=3, color="red")

            ax.add_patch(rect)

    # Annotate agents
    for i, agent_track in enumerate(scenario.tracks):

        state_step = agent_track.states[step]

        annotate_id = f"{agent_track.id}"
        speed = get_agent_speed(agent_track, step)
        annotate_id_speed = f"{agent_track.id}: {speed}"

        annotation: str = annotate_id

        if annotate_type == AnnotateType.WITH_ID_AND_SPEED:
            annotation = annotate_id_speed

        if agent_track.id in agents_in_intersection:
            annotation = f"{annotation} ⚑"

        if agent_track.states[step].valid and is_agent_in_lane_change_intersection_at_tick(scenario, lane_changes, step,
                                                                                           agent_track.id, kd_tree):
            annotation = f"{annotation} ⚑L"

        list_agent_labels = next((al for idx, al in agents_labels if idx == agent_track.id), [])
        agent_labels = next(((st, e1, e2) for st, e1, e2 in list_agent_labels if st == step), None)

        if agent_labels:
            st, priority_labels, step_states = agent_labels
            direction_label, speed_label = priority_labels

            annotation = f"{annotation}\n{direction_label.value},{speed_label.value}"

        bbox = dict(boxstyle="round", fc="0.8")
        agent_type = agent_track.object_type

        # If agent is pedestrian or cyclist, use different color for annotation box
        if agent_type == Track.ObjectType.TYPE_PEDESTRIAN or agent_type == Track.ObjectType.TYPE_CYCLIST:
            bbox["fc"] = COLOR_PEDESTRIAN_CYCLIST
        elif i == scenario.sdc_track_index:
            bbox["fc"] = COLOR_AUTONOMOUS_VEHICLE

        ax.annotate(
            annotation,
            (state_step.center_x, state_step.center_y),
            bbox=bbox,
            ha="center",
            va="center",
            fontsize=ANNOTATE_TEXT_SIZE
        )


def visualize_traffic_lights(scenario: Scenario, ax, step):
    traffic_signals_y = []
    traffic_signals_x = []
    traffic_signals_color = []

    for lane_state in scenario.dynamic_map_states[step].lane_states:
        traffic_signals_y.append(lane_state.stop_point.y)
        traffic_signals_x.append(lane_state.stop_point.x)
        signal_state = TrafficSignalLaneState.State

        if lane_state.state == signal_state.LANE_STATE_UNKNOWN:

            traffic_signals_color.append(TRAFFIC_SIGNAL_COLOR_STATE_UNKNOWN)
        elif lane_state.state in [signal_state.LANE_STATE_ARROW_GO,
                                  signal_state.LANE_STATE_GO]:

            traffic_signals_color.append(TRAFFIC_SIGNAL_COLOR_STATE_GO)
        elif lane_state.state in [signal_state.LANE_STATE_ARROW_CAUTION,
                                  signal_state.LANE_STATE_CAUTION,
                                  signal_state.LANE_STATE_FLASHING_CAUTION]:

            traffic_signals_color.append(TRAFFIC_SIGNAL_COLOR_STATE_CAUTION)
        elif lane_state.state in [signal_state.LANE_STATE_ARROW_STOP,
                                  signal_state.LANE_STATE_STOP,
                                  signal_state.LANE_STATE_FLASHING_STOP]:

            traffic_signals_color.append(TRAFFIC_SIGNAL_COLOR_STATE_STOP)

    ax.scatter(traffic_signals_x, traffic_signals_y, marker='o', alpha=1, s=8, color=traffic_signals_color)


def visualize_crosswalks(scenario: Scenario, ax, white=False):
    crosswalk_features = [map_feature for map_feature in scenario.map_features
                          if map_feature.WhichOneof("feature_data") == "crosswalk"]

    crosswalks = []

    for crosswalk_feature in crosswalk_features:
        feature = crosswalk_feature.crosswalk
        polygon = []

        for polyline in feature.polygon:
            polygon.append((polyline.x, polyline.y))

        crosswalks.append(polygon)

    for crosswalk in crosswalks:
        if white:
            poly = plt.Polygon(crosswalk, color="black", alpha=0.1)
        else:
            poly = plt.Polygon(crosswalk, color="white", alpha=0.1)
        ax.add_patch(poly)


def visualize_road_center_ids(scenario: Scenario, ax):
    road_center_id = []

    road_center_features = [map_feature for map_feature in scenario.map_features
                            if map_feature.WhichOneof("feature_data") == "lane"]

    for road_center_feature in road_center_features:
        feature_data_type = road_center_feature.WhichOneof("feature_data")
        feature = getattr(road_center_feature, feature_data_type)
        index = math.floor(len(feature.polyline) / 2)

        road_center_id.append(
            ((feature.polyline[index].x, feature.polyline[index].y), road_center_feature.id)
        )

    for road_center in road_center_id:
        pos, id = road_center

        bbox = dict(boxstyle="round", fc="0.8")
        ax.annotate(
            id,
            pos,
            bbox=bbox,
            ha="center",
            va="center",
            fontsize=5
        )


def visualize_road_center_neighbors_ids(scenario: Scenario, ax):
    road_neighbors = []

    road_center_features = [map_feature for map_feature in scenario.map_features
                            if map_feature.WhichOneof("feature_data") == "lane"]

    for road_center_feature in road_center_features:
        road_center_feature: LaneCenter
        feature_data_type = road_center_feature.WhichOneof("feature_data")
        feature = getattr(road_center_feature, feature_data_type)
        index = math.floor(len(feature.polyline) / 2)

        s = f"{road_center_feature.id}\nL: "
        for lane_neighbor in feature.left_neighbors:
            lane_neighbor: LaneNeighbor
            s = f"{s},{lane_neighbor.feature_id}"

        s = f"{s} \nR: "
        for lane_neighbor in feature.right_neighbors:
            lane_neighbor: LaneNeighbor
            s = f"{s},{lane_neighbor.feature_id}"

        road_neighbors.append(
            ((feature.polyline[index].x, feature.polyline[index].y), s)
        )

    for road_neighbor in road_neighbors:
        pos, string = road_neighbor

        bbox = dict(boxstyle="round", fc="0.8")
        ax.annotate(
            string,
            pos,
            bbox=bbox,
            ha="center",
            va="center",
            fontsize=3
        )


def visualize_intersection(scenario: Scenario, ax):
    intersections, lane_changes = get_intersection_lane_changes_cache(scenario)

    filtered_intersections = [intersection for intersection in intersections if len(intersection) > 1]

    for intersection in filtered_intersections:
        road_centers: list[LaneCenter] = [getattr(map_feature, "lane") for map_feature in scenario.map_features if
                                          map_feature.id in intersection]
        points = []

        for road_center in road_centers:
            points.append((road_center.polyline[0].x, road_center.polyline[0].y))
            points.append((road_center.polyline[len(road_center.polyline) - 1].x,
                           road_center.polyline[len(road_center.polyline) - 1].y))

        poly = plt.Polygon(ccw_sort(points), color="sandybrown", alpha=0.1)
        ax.add_patch(poly)

    for lane_change in lane_changes:
        road_centers: list[LaneCenter] = [getattr(map_feature, "lane") for map_feature in scenario.map_features if
                                          map_feature.id in lane_change]
        num_points = min([len(road_center.polyline) for road_center in road_centers])
        points = []

        for road_center in road_centers:
            points.append((road_center.polyline[0].x, road_center.polyline[0].y))
            points.append((road_center.polyline[num_points - 1].x, road_center.polyline[num_points - 1].y))

        poly = plt.Polygon(ccw_sort(points), color="lime", alpha=0.1)
        ax.add_patch(poly)


def visualize_road_type(scenario: Scenario, ax):
    road_types = []
    road_center_y = []
    road_center_x = []

    road_center_features = [map_feature for map_feature in scenario.map_features
                            if map_feature.WhichOneof("feature_data") == "lane"]

    for road_center_feature in road_center_features:
        road_center_feature: LaneCenter
        feature_data_type = road_center_feature.WhichOneof("feature_data")
        feature = getattr(road_center_feature, feature_data_type)
        index = math.floor(len(feature.polyline) / 2)

        s = "none"

        # do not check crossings for bike lanes
        if feature.type != 3:
            feature_coords = road_center_feature.lane
            road_center_y.extend([polyline.y for polyline in feature_coords.polyline])
            road_center_x.extend([polyline.x for polyline in feature_coords.polyline])

            current_y = [polyline.y for polyline in feature_coords.polyline]
            current_x = [polyline.x for polyline in feature_coords.polyline]
            line_outer_start = [current_x[0], current_y[0]]
            line_outer_end = [current_x[len(current_x) - 1], current_y[len(current_y) - 1]]

            for road_center_feature_inner in road_center_features:
                # no need to check further if a crossing has been found
                if s == "crossing":
                    break
                road_center_feature_inner: LaneCenter
                feature_data_type_inner = road_center_feature_inner.WhichOneof("feature_data")
                feature_inner = getattr(road_center_feature_inner, feature_data_type_inner)

                # do not count crossing with bike lane
                if feature_inner.type == 3:
                    continue

                feature_coords_inner = road_center_feature_inner.lane
                current_y = [polyline.y for polyline in feature_coords_inner.polyline]
                current_x = [polyline.x for polyline in feature_coords_inner.polyline]
                line_inner_start = [current_x[0], current_y[0]]
                line_inner_end = [current_x[len(current_x) - 1], current_y[len(current_y) - 1]]

                intersection = maths.intersection(line_outer_start, line_outer_end, line_inner_start, line_inner_end)
                if intersection > 0:
                    s = "lane change"
                    for lane_neighbor in feature.left_neighbors:
                        if lane_neighbor.feature_id - road_center_feature_inner.id != 0:
                            s = "crossing"
                    for lane_neighbor in feature.right_neighbors:
                        if lane_neighbor.feature_id - road_center_feature_inner.id != 0:
                            s = "crossing"

        s = f"{road_center_feature.id}\n{s}"

        road_types.append(
            ((feature.polyline[index].x, feature.polyline[index].y), s)
        )

    for road_neighbor in road_types:
        pos, string = road_neighbor

        bbox = dict(boxstyle="round", fc="0.8")
        ax.annotate(
            string,
            pos,
            bbox=bbox,
            ha="center",
            va="center",
            fontsize=4
        )


def visualize_road_edge_ids(scenario: Scenario, ax):
    road_line_edge_id = []

    road_lane_types = ["road_line", "road_edge"]

    road_line_edge_features = [map_feature for map_feature in scenario.map_features
                               if map_feature.WhichOneof("feature_data") in road_lane_types]

    for road_line_edge_feature in road_line_edge_features:
        feature_data_type = road_line_edge_feature.WhichOneof("feature_data")
        feature = getattr(road_line_edge_feature, feature_data_type)
        index = math.floor(len(feature.polyline) / 2)
        road_line_edge_id.append(
            ((feature.polyline[index].x, feature.polyline[index].y), road_line_edge_feature.id)
        )

    for road_edge in road_line_edge_id:
        pos, id = road_edge

        bbox = dict(boxstyle="round", fc="tan")
        ax.annotate(
            id,
            pos,
            bbox=bbox,
            ha="center",
            va="center",
            fontsize=5
        )


def is_sufficient_luminance_diff(color_a, color_b):
    r_a, g_a, b_a, _ = color_a
    r_b, g_b, b_b, _ = color_b
    lum_a = 0.3 * r_a + 0.59 * g_a + 0.11 * b_a
    lum_b = 0.3 * r_b + 0.59 * g_b + 0.11 * b_b

    return abs(lum_a - lum_b) > 0.3


def get_colors_map(indices, white=False):
    num_indices = len(indices)
    if white:
        c_list = get_color_list(num_indices, num_indices, (1, 1, 1, 1))
    else:
        c_list = get_color_list(num_indices, num_indices, (0, 0, 0, 1))
    c_map = {}
    for i in range(0, num_indices):
        c_map[indices[i]] = c_list[i]

    return c_map


def get_color_list(size: int, original_size, background):
    colors = plt.cm.get_cmap('jet', size)
    colors = colors(range(size))

    num_insufficient_luminance = 0

    for i, color in enumerate(colors):
        if not is_sufficient_luminance_diff(color, background):
            num_insufficient_luminance += 1
            colors = [color for j, color in enumerate(colors) if j != i]

    if len(colors) < original_size:
        return get_color_list(size + num_insufficient_luminance, original_size, background)

    random.Random(COLOR_MAP_SEED).shuffle(colors)

    return colors
