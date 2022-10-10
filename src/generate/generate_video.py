import concurrent.futures
import uuid
from moviepy.editor import *
from config import *
from matplotlib import pyplot as plt, animation
from pathlib import Path


def create_animation(images):
    """ Creates a Matplotlib animation of the given images.

  Args:
    images: A list of numpy arrays representing the images.

  Returns:
    A matplotlib.animation.Animation.

  Usage:
    anim = create_animation(images)
    anim.save('/tmp/animation.avi')
    HTML(anim.to_html5_video())
  """

    plt.ioff()
    fig, ax = plt.subplots()
    dpi = DPI
    size_inches = dpi / 10
    fig.set_size_inches([size_inches, size_inches])
    fig.set_facecolor('black')

    def animate_func(i):
        ax.imshow(images[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('black')
        ax.grid('off')

    anim = animation.FuncAnimation(
        fig, animate_func, frames=len(images), interval=100)
    plt.close(fig)
    return anim


def fast_create_video(images: list, folder=f"{OUTPUT_DIR}/str(int(time.time()))", video_name="video"):
    """
    Creates a video for a scene
    Args:
        images: The frames generated with matplotlib
        folder: Destination folder of the video
        video_name: Video file name
    """
    image_chunks = list(split(images, NUM_THREADS))
    rand_uid = uuid.uuid4()

    path = f'{PROJECT_DIR}/{folder}'
    Path(path).mkdir(parents=True, exist_ok=True)

    clips: list[VideoFileClip]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(create_and_save_video, chunk, i, path, rand_uid) for i, chunk in enumerate(image_chunks)]
        clips = [f.result() for f in futures]

    video = concatenate_videoclips(clips, method='compose')
    video.write_videofile(f'{path}/{video_name}.mp4', verbose=False, logger=None)

    for clip in clips:
        os.remove(clip.filename)


def create_and_save_video(image_chunk: list, index, path, uid):
    anim = create_animation(image_chunk)
    dir = f'{path}/split_{index}_{uid}.mp4'
    anim.save(dir, fps=VIDEO_FPS)
    video = VideoFileClip(dir)

    return video


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

