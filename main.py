import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from PIL import Image
import imageio.v3 as iio
import torch
from new_pipe import StableVideoDiffusionPipeline

first_img = Image.open("first_frame.png")
frames = iio.imread("street.mp4", plugin="pyav")
frames = [Image.fromarray(frame).resize((640, 640)) for frame in frames ][:25]
frames = (frames + frames)[:25]
width, height = frames[0].size
first_img.resize((width,height))

print(width
      ,height)
device = "cuda"
noise_aug_strength = 0.02
num_inference_steps = 20

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    # "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
    # torch_dtype=torch.float16,
    variant="fp16"
).to(device)

pipe.scheduler.set_timesteps(num_inference_steps, device=device)

generator = torch.cuda.manual_seed(42)

latents = []
with torch.inference_mode():
    # breakpoint()
    for frame in frames:
        image = pipe.image_processor.preprocess(frame, height=height, width=width).to(device)
        
        needs_upcasting = pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast
        if needs_upcasting:
            pipe.vae.to(dtype=torch.float32)

        image_latents = pipe._encode_vae_image(
                image,
                device=device,
                num_videos_per_prompt=1,
                # ??
                do_classifier_free_guidance=False,
            )
        
        latents.append(image_latents)

latents = torch.stack(latents)
# latents = latents.to(torch.float16)
latents = latents.transpose(0,1)
latents = pipe.vae.config.scaling_factor * latents

print(latents.shape)
# pipe.enable_model_cpu_offload()

frames = pipe(
    first_img,
    decode_chunk_size=8,
    generator=generator,
    width=width,
    height=height,
    latents=latents,
    denoising_strength=0.2
)
frames_list = sum(frames.frames, [])
export_to_video(frames_list, "output_video.mp4", fps=7)