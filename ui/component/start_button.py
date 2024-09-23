import streamlit as st
from manage.thread_manage import ThreadManager
from manage.model_manage import load_model, download_model, prepare_model
import cv2

def kill_model():
    if st.session_state.model is not None:
        del st.session_state.model
        st.session_state.model = None

def kill_model_thread(thread_var: ThreadManager):
    if thread_var.thread is not None:
        thread_var.thread_enabled = False
        thread_var.thread.join()
        thread_var.thread = None

    capture = st.session_state.video_capture
    if capture is not None:
        capture.release()
        st.session_state.video_capture = None



# Execute model
def start_process(thread_var: ThreadManager, model_type, weight_path):
    

    global g_frame_queue, g_frame_queue_updated, g_out_sim_scores, g_out_sim_softmax, g_text_vectors
    global g_thread_enabled, g_thread_lock


    # check video
    video_path = st.session_state.video_path
    if not video_path:
        print('error: video is not uploaded')
    
    # kill thread if running
    if thread_var.thread_enabled or st.session_state.model_thread is not None:
        kill_model_thread(thread_var)

    # kill model if changed
    if model_type != st.session_state.model_type or weight_path != st.session_state.model_weight_path:
        st.session_state.model_type = model_type
        st.session_state.model_weight_path = weight_path
        kill_model(thread_var)

    # load model
    model = prepare_model()
    # encode all prompts and save
    texts = st.session_state.prompt_all_text_list
    if texts:
        thread_var.text_vectors = thread_var.model.encode_texts_to_vector(texts)

    # get video settings
    frame_len = st.session_state.frame_len
    frame_int = st.session_state.frame_int
    tile_size = st.session_state.tile_size
    tile_margin = st.session_state.tile_margin

    # make Video capture
    video_path = st.session_state.video_path
    delay_state = not isinstance(video_path, str) or not video_path.startswith("rtsp")

    video_capture = cv2.VideoCapture(video_path)
    st.session_state.video_capture = video_capture

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    counter = 0

    # start model thread
    start_model_thread(thread_var)

    # 비디오 저장을 위한 queue 생성
    video_record_queue = Queue(maxsize=100)

    # 비디오 저장 쓰레드 생성
    record_dir_path = st.session_state.save_path
    video_record_thread = threading.Thread(
        target=record_video,
        args=(video_record_queue, record_dir_path, fps),
        daemon=True
    )
    video_record_thread.start()

    # main loop
    try:
        while True:
            res, frame = video_capture.read()
            if not res:
                print("stream terminated. escaping...")
                break

            # put frame to queue if path is available
            if record_dir_path:
                video_record_queue.put(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            tiles_comb = None
            if counter % int(fps * frame_int) == 0:
                tiles, wn, hn = preprocess_input_image(thread_var.model, origin_image=frame, tile_size=tile_size, margin_size=tile_margin)

                if wn > 1 or hn > 1:
                    tiles_comb = combine_image_tiles(tiles, wn, hn)
                else:
                    tiles_comb = frame

                st.session_state.tiled_images = tiles

                with thread_var.thread_lock:
                    if len(thread_var.frame_queue) != frame_len:
                        mask = np.zeros_like(tiles)
                        for _ in range(frame_len):
                            thread_var.add_frame(mask)
                    thread_var.add_frame(tiles)

            # update output ui - current frame
            source_frame = st.session_state.source_frame_com

            # Resize frame
            frame = cv2.resize(frame, (MAX_WIDTH, MAX_HEIGHT), interpolation=cv2.INTER_LINEAR)
            source_frame.image(frame, use_column_width=True)

            # tiled
            if tiles_comb is not None:
                tiled_frame = st.session_state.tiled_frame_com
                tiles_comb = cv2.resize(tiles_comb, (MAX_WIDTH, MAX_HEIGHT), interpolation=cv2.INTER_LINEAR)
                tiled_frame.image(tiles_comb, use_column_width=True)

            # 값에 변동 없으면 update 제한
            previous_score = st.session_state.previous_score
            if not np.array_equal(previous_score, thread_var.out_sim_scores):

                # make text output
                make_text_output(thread_var.out_sim_scores)

                # update graph
                update_graph(thread_var.out_sim_scores)

            st.session_state.previous_score = thread_var.out_sim_scores

            if delay_state:     # 파일 읽어올 때는 딜레이 설정
                time.sleep(1.0 / fps - 0.01)

            counter += 1

    except Exception as e:
        traceback.print_exc()

    finally:
        video_capture.release()

def start_btn():
    model_type = st.session_state.model_type
    model_weight_path = st.session_state.model_weight_path

    if st.button("Start", key="st_btn_run", type="primary", disabled=st.session_state.thread_enabled):
        # Initializing the session state
        st.session_state.prompt_score_dict = {}

        # Start process
        thread_var = ThreadManager()
        start_process(thread_var, model_type, model_weight_path)

