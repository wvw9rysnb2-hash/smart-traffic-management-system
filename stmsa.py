import streamlit as st
import cv2
import tempfile
import os
import pandas as pd
import plotly.express as px
from collections import defaultdict
from ultralytics import YOLO

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Smart Traffic Management System",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Smart Traffic Management System")
st.write("Upload a traffic video to count unique vehicles.")

# ----------------------------
# LOAD YOLO MODEL
# ----------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load YOLO model: {e}")
    st.stop()

# ----------------------------
# FILE UPLOAD
# ----------------------------
uploaded_video = st.file_uploader(
    "Upload Traffic Video",
    type=["mp4", "avi", "mov", "mkv"]
)

# ----------------------------
# PROCESS VIDEO
# ----------------------------
if uploaded_video is not None:

    st.video(uploaded_video)

    if st.button("Start Vehicle Detection", type="primary"):

        temp_path = None
        cap = None

        try:
            # Save uploaded file
            suffix = os.path.splitext(uploaded_video.name)[1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:

                temp_file.write(uploaded_video.getbuffer())
                temp_path = temp_file.name

            cap = cv2.VideoCapture(temp_path)

            if not cap.isOpened():
                raise RuntimeError(
                    "Could not open uploaded video."
                )

            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            progress_bar = st.progress(0)
            status_text = st.empty()

            # COCO Classes
            class_to_category = {
                1: "Bikes",   # Bicycle
                2: "Cars",    # Car
                3: "Bikes",   # Motorcycle
                5: "Buses",   # Bus
                7: "Trucks"   # Truck
            }

            counts = defaultdict(int)
            seen_tracks = set()

            frame_number = 0

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[1, 2, 3, 5, 7],
                    conf=0.25,
                    verbose=False
                )

                result = results[0]

                if (
                    result.boxes is not None
                    and result.boxes.id is not None
                ):

                    track_ids = (
                        result.boxes.id.int()
                        .cpu()
                        .tolist()
                    )

                    class_ids = (
                        result.boxes.cls.int()
                        .cpu()
                        .tolist()
                    )

                    for track_id, class_id in zip(
                        track_ids,
                        class_ids
                    ):

                        if track_id not in seen_tracks:

                            category = (
                                class_to_category.get(class_id)
                            )

                            if category:

                                seen_tracks.add(track_id)
                                counts[category] += 1

                frame_number += 1

                if total_frames > 0:
                    progress_bar.progress(
                        min(
                            frame_number / total_frames,
                            1.0
                        )
                    )

                if frame_number % 30 == 0:
                    status_text.write(
                        f"Processed {frame_number:,} frames..."
                    )

            cap.release()
            cap = None

            progress_bar.progress(1.0)
            status_text.success(
                "✅ Video processing complete!"
            )

            # ----------------------------
            # METRICS
            # ----------------------------
            st.subheader("📊 Unique Vehicle Counts")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "🚗 Cars",
                counts["Cars"]
            )

            col2.metric(
                "🚌 Buses",
                counts["Buses"]
            )

            col3.metric(
                "🚚 Trucks",
                counts["Trucks"]
            )

            col4.metric(
                "🏍️ Bikes",
                counts["Bikes"]
            )

            total = sum(counts.values())

            st.metric(
                "🚦 Total Unique Vehicles",
                total
            )

            # ----------------------------
            # DATAFRAME
            # ----------------------------
            vehicle_data = pd.DataFrame({

                "Vehicle Type": [
                    "Cars",
                    "Buses",
                    "Trucks",
                    "Bikes"
                ],

                "Count": [
                    counts["Cars"],
                    counts["Buses"],
                    counts["Trucks"],
                    counts["Bikes"]
                ]
            })

            st.divider()

            st.subheader(
                "📈 Traffic Analysis Dashboard"
            )

            col_left, col_right = st.columns(2)

            # ----------------------------
            # BAR CHART
            # ----------------------------
            with col_left:

                fig_bar = px.bar(
                    vehicle_data,
                    x="Vehicle Type",
                    y="Count",
                    text="Count",
                    title="Vehicle Count Comparison"
                )

                fig_bar.update_layout(
                    showlegend=False,
                    xaxis_title="Vehicle Category",
                    yaxis_title="Count"
                )

                st.plotly_chart(
                    fig_bar,
                    use_container_width=True
                )

            # ----------------------------
            # PIE CHART
            # ----------------------------
            with col_right:

                if total > 0:

                    fig_pie = px.pie(
                        vehicle_data,
                        names="Vehicle Type",
                        values="Count",
                        hole=0.4,
                        title="Vehicle Distribution"
                    )

                    fig_pie.update_traces(
                        textinfo="label+percent"
                    )

                    st.plotly_chart(
                        fig_pie,
                        use_container_width=True
                    )

                else:
                    st.info(
                        "No vehicles detected."
                    )

            st.divider()

            # ----------------------------
            # TABLE
            # ----------------------------
            st.subheader(
                "Vehicle Count Summary"
            )

            st.dataframe(
                vehicle_data,
                use_container_width=True
            )

            st.caption(
                "Bikes combines bicycles and motorcycles. "
                "Each vehicle is counted once using its tracker ID."
            )

            if total == 0:
                st.warning(
                    "No vehicles detected. Try a clearer video."
                )

        except Exception as e:

            st.error(
                f"Detection failed: {str(e)}"
            )

        finally:

            if cap is not None:
                cap.release()

            if (
                temp_path
                and os.path.exists(temp_path)
            ):
                os.remove(temp_path)
