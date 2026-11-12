
import streamlit as st
import cv2
import tempfile
import os
import pandas as pd
import plotly.express as px
from collections import defaultdict
from ultralytics import YOLO

st.set_page_config(
    page_title="Smart Traffic Management System",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Smart Traffic Management System")
st.write("Upload a traffic video to count unique vehicles.")


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


model = load_model()

uploaded_video = st.file_uploader(
    "Upload traffic video",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_video is not None:

    st.video(uploaded_video)

    if st.button("Start Vehicle Detection", type="primary"):

        temp_path = None
        cap = None

        try:
            # Save uploaded video temporarily
            suffix = os.path.splitext(uploaded_video.name)[1]

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as temp_file:
                temp_file.write(uploaded_video.getbuffer())
                temp_path = temp_file.name

            cap = cv2.VideoCapture(temp_path)

            if not cap.isOpened():
                raise RuntimeError("Could not open uploaded video.")

            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            progress = st.progress(0)
            status = st.empty()

            # COCO vehicle class IDs
            class_to_category = {
                1: "Bikes",
                2: "Cars",
                3: "Bikes",
                5: "Buses",
                7: "Trucks"
            }

            # Track each vehicle only once
            seen_tracks = set()
            counts = defaultdict(int)

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
                    verbose=False,
                    conf=0.25
                )

                result = results[0]

                if (
                    result.boxes is not None
                    and result.boxes.id is not None
                ):
                    track_ids = (
                        result.boxes.id.int().cpu().tolist()
                    )

                    class_ids = (
                        result.boxes.cls.int().cpu().tolist()
                    )

                    for track_id, class_id in zip(
                        track_ids, class_ids
                    ):

                        if track_id not in seen_tracks:

                            category = class_to_category.get(
                                class_id
                            )

                            if category is not None:
                                seen_tracks.add(track_id)
                                counts[category] += 1

                frame_number += 1

                if total_frames > 0:
                    progress.progress(
                        min(frame_number / total_frames, 1.0)
                    )

                if frame_number % 30 == 0:
                    status.write(
                        f"Processed {frame_number} frames..."
                    )

            cap.release()
            cap = None

            progress.progress(1.0)
            status.success("Video processing complete!")

            # VEHICLE COUNTS
            st.subheader("📊 Unique Vehicle Counts")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("🚗 Cars", counts["Cars"])
            col2.metric("🚌 Buses", counts["Buses"])
            col3.metric("🚚 Trucks", counts["Trucks"])
            col4.metric("🏍️ Bikes", counts["Bikes"])

            total = sum(counts.values())

            st.metric("Total Unique Vehicles", total)

            # PREPARE DATA
            vehicle_data = pd.DataFrame({
                "Vehicle Type": [
                    "Cars", "Buses", "Trucks", "Bikes"
                ],
                "Count": [
                    counts["Cars"],
                    counts["Buses"],
                    counts["Trucks"],
                    counts["Bikes"]
                ]
            })

            st.divider()

            st.subheader("📊 Traffic Analysis Dashboard")

            col1, col2 = st.columns(2)

            # BAR CHART
            with col1:

                st.write("### Vehicle Count Comparison")

                fig_bar = px.bar(
                    vehicle_data,
                    x="Vehicle Type",
                    y="Count",
                    text="Count",
                    title="Number of Vehicles by Type"
                )

                fig_bar.update_layout(
                    xaxis_title="Vehicle Category",
                    yaxis_title="Number of Vehicles",
                    showlegend=False
                )

                st.plotly_chart(
                    fig_bar,
                    use_container_width=True
                )

            # PIE CHART
            with col2:

                st.write("### Vehicle Distribution")

                if total > 0:

                    fig_pie = px.pie(
                        vehicle_data,
                        names="Vehicle Type",
                        values="Count",
                        hole=0.4,
                        title="Percentage of Each Vehicle Type"
                    )

                    fig_pie.update_traces(
                        textinfo="label+percent"
                    )

                    st.plotly_chart(
                        fig_pie,
                        use_container_width=True
                    )

                else:
                    st.info("No vehicles detected to display.")

            st.divider()

            # SUMMARY TABLE
            st.subheader("Vehicle Count Summary")

            st.dataframe(
                vehicle_data,
                use_container_width=True
            )

            st.caption(
                "Bikes combines bicycles and motorcycles. "
                "Vehicles are counted by tracking ID, not by "
                "the number of frames in which they appear."
            )

            if total == 0:
                st.warning(
                    "No vehicles were tracked. Try a clearer "
                    "video or check that the model loaded correctly."
                )

        except Exception as e:
            st.error(f"Detection failed: {e}")

        finally:
            if cap is not None:
                cap.release()

            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
