import gradio as gr
import os
import subprocess
import argparse

REPO_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(REPO_DIR, "data")
OUTPUT_DIR = os.path.join(REPO_DIR, "output")

def extract_frames(start_time, dur_time, fps, video_file):
    # 1. Check if data directory exists, if not, create it
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # 2. Create a directory with the same name as the video file in the data directory
    video_name = os.path.splitext(video_file.name)[0]
    video_dir = os.path.join(DATA_DIR, video_name)

    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    # Create an input directory inside the video directory
    input_dir = os.path.join(video_dir, "input")
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

    # 3. Save the video file to the video directory
    video_path = os.path.join(video_dir, video_file.name)
    video_file.save(video_path)

    # 4. Run ffmpeg to process the video
    output_path = os.path.join(input_dir, "%04d.jpg")
    command = f'ffmpeg -ss {start_time} -t {dur_time} -i {video_path} -vf "fps={fps}" -q:v 1 "{output_path}"'
    subprocess.run(command, shell=True, check=True)
    
    return video_dir, "Frames extracted successfully."

def run_colmap_and_train(video_dir):
    commands = [
        f'python convert.py -s "{video_dir}"',
        f'python train.py -s "{video_dir}"'
    ]
    for command in commands:
        subprocess.run(command, shell=True, check=True)
    return "Training completed successfully."

def create_ui():
        with gr.Blocks(analytics_enabled=False) as ui_component:
            with gr.Accordion("Video Input", open=True):
                video_input = gr.Video(label="Upload Video")
                start_time = gr.components.Textbox(lines=1, placeholder="00:00:10", label="Start Time")
                dur_time = gr.components.Textbox(lines=1, placeholder="00:00:50", label="Duration")
                frames_per_second = gr.components.Textbox(lines=1, placeholder="2", label="Frames per Second")
                btn_extract = gr.Button("Extract Frames", variant="primary")

            with gr.Accordion("Train", open=True):
                btn_compute = gr.Button("Compute", variant="primary")

            output_text = gr.components.Textbox(lines=1, label="Output")

            btn_extract.click(
                fn=extract_frames,
                inputs=[video_input, start_time, dur_time, frames_per_second],
                outputs=[output_text]
            )

            btn_compute.click(
                fn=run_colmap_and_train,
                inputs=[],
                outputs=[output_text]
            )

            txtbox_lora_output_dir.change(
                fn=partial(modify_general_parameter, the_elem_id=txtbox_lora_output_dir.elem_id),
                inputs=[txtbox_lora_output_dir, combo_lora_subtask_list],
                outputs=[]
            )

        return ui_component

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_name", default="localhost", help="Server name for the Gradio interface")
    parser.add_argument("--server_port", type=int, default=7860, help="Server port for the Gradio interface")
    args = parser.parse_args()

    create_ui().launch(server_name=args.server_name, server_port=args.server_port, debug=True)