from __future__ import annotations


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
from pathlib import Path
from pyremesh import remesh_botsch
import torch
from tqdm import tqdm

from nds.core import (
    Mesh, Renderer
)
from nds.losses import (
    mask_loss, normal_consistency_loss, laplacian_loss, shading_loss
)
from nds.modules import (
    SpaceNormalization, NeuralShader, EditViewSampler
)
from nds.utils import (
    AABB, read_views, read_mesh, write_mesh, visualize_mesh_as_overlay, visualize_views, generate_mesh, mesh_generator_names
)

from einops import rearrange
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from PIL import Image
from ip2p import MultiControlNet, ControlNet
import yaml, shutil, os

CONTROLNET_NORMAL_SOURCE = "lllyasviel/control_v11p_sd15_normalbae"
CONTROLNET_IP2P_SOURCE = "lllyasviel/control_v11e_sd15_ip2p"

SD_SOURCE = "runwayml/stable-diffusion-v1-5"
import yaml


def blackbox_update_view(original_image, rendered_image, pipe, args, normal_image, iteration, path, noise = None):
    original_image = original_image.unsqueeze(0).permute(0, 3, 1, 2).to(torch.float16)
    rendered_image = rendered_image.unsqueeze(0).permute(0, 3, 1, 2).to(torch.float16)
    normal_image = normal_image.unsqueeze(0).permute(0, 3, 1, 2).to(torch.float16)
    
    prompt = args.prompt
    negative_prompt = args.negative_prompt
    images = [original_image, normal_image]
    # rendered_image.save(shaded_path / f'rendered_{iteration}.png')
    #shaded_path = (images_save_path / "shaded"/ f'neuralshading_{iteration}.png' )
    #normal_path = (images_save_path / "normal"/ f'neuralshading_{iteration}.png' )
    #rendered_image = Image.open(shaded_path)
    #normal_image = Image.open(normal_path)
    #images = [rendered_image, normal_image]
    generator = torch.Generator(device="cpu").manual_seed(42)
    edited_image = pipe(
        prompt,
        images,
        num_inference_steps=10,
        generator=generator,
        #negative_prompt=negative_prompt,
        controlnet_conditioning_scale=args.conditoning_scale,
    ).images[0]
    
    if (iteration % args.edit_save_frequency == 0):
        saved_image_path = str(edits_save_path)+"/"+str(iteration)+".png"
        edited_image.save(saved_image_path)
    edited_image = torch.tensor(np.array(edited_image), device = original_image.device) / 255.0

    return edited_image

def our_update_view(original_image, rendered_image, model, args, normal_image, iteration, path, noise = None, latent_weight = None, depth_image = None):
    original_image = original_image.unsqueeze(0).permute(0, 3, 1, 2)
    rendered_image = rendered_image.unsqueeze(0).permute(0, 3, 1, 2)
    normal_image = normal_image.unsqueeze(0).permute(0, 3, 1, 2)
    
    #conditions = [original_image,normal_image]
    if depth_image != None:
        depth_image = depth_image.unsqueeze(0).permute(0, 3, 1, 2)
        #conditions.append(depth_image)
    if not args.ip2p_use_full_precision:
        original_image = original_image.to(torch.float16)
        rendered_image = rendered_image.to(torch.float16)
        normal_image = normal_image.to(torch.float16)
        if depth_image != None:
            depth_image = depth_image.to(torch.float16)
    ip2p_device = model.device
    
    if not latent_weight:
        latent_weight = args.latent_weight
    
    prompt = args.prompt
    guidance_scale, diffusion_steps = args.cfg_text, args.steps


    text_embedding = model.pipe._encode_prompt(
            prompt, device= ip2p_device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=args.negative_prompt
        )
    
    text_embedding = text_embedding.to(original_image.dtype)#for full precision

    edited_image = model.edit_image(
                            text_embedding.to(ip2p_device),
                            rendered_image,
                            original_image,
                            normal_image,
                            depth_image,
                            guidance_scale = guidance_scale,
                            diffusion_steps = diffusion_steps,
                            noise = noise,
                            latent_weight = latent_weight,
                        )
        
    edited_image = rearrange(edited_image, "1 c h w -> h w c")
    if (iteration % args.edit_save_frequency == 0):
        edit_x = 255.0 * edited_image
        saved_image = Image.fromarray(edit_x.type(torch.uint8).cpu().numpy())
        saved_image_path = str(edits_save_path)+"/"+str(iteration)+".png"
        saved_image.save(saved_image_path)
    return edited_image

def update_gt_views(views, mesh, renderer):
    with torch.no_grad():
        for i, view in enumerate(views):
            gbuffer = renderer.render([view], mesh, channels=['mask', 'position', 'normal'], with_antialiasing=True)[0]
            #dataset update
            position, normal = gbuffer["position"], gbuffer["normal"]
            view_direction =  torch.nn.functional.normalize(view.camera.center - position, dim=-1)
            views[i].color = shader(position, normal, view_direction, iteration) * gbuffer["mask"] + (1-gbuffer["mask"])
            rgb_threshold = torch.tensor([0.95, 0.95, 0.95], device = views[i].color.device, dtype = views[i].color.dtype)
            views[i].mask = torch.any(views[i].color <= rgb_threshold, dim=-1).unsqueeze(-1).float()
            # shaded_path = images_save_path / "loss"
            # shaded_path.mkdir(parents=True, exist_ok=True)
            # plt.imsave(shaded_path / f'rendered_{iteration}_{i}.png', views[i].color.detach().cpu().numpy())
            # print("updated views")
    print("updated views")

if __name__ == '__main__':
    parser = ArgumentParser(description='Multi-View Mesh Reconstruction with Neural Deferred Shading', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default=None, help='Path to YAML configuration file')


    parser.add_argument('--input_dir', type=Path, default="./data/122_owl_resized/views", help="Path to the input data")
    parser.add_argument('--input_bbox', type=Path, default="./data/122_owl/bbox.txt", help="Path to the input bounding box. If None, it is computed from the input mesh")
    parser.add_argument('--output_dir', type=Path, default="./out", help="Path to the output directory")
    parser.add_argument('--experiment_name', type=Path, default="experiment", help="Name of this run")
    parser.add_argument('--initial_mesh', type=str, default="vh32", help="Initial mesh, either a path or one of [vh16, vh32, vh64, sphere16]")
    parser.add_argument('--image_scale', type=int, default=1, help="Scale applied to the input images. The factor is 1/image_scale, so image_scale=2 halves the image size")
    parser.add_argument('--iterations', type=int, default=2000, help="Total number of iterations")
    parser.add_argument('--lr_vertices', type=float, default=1e-3, help="Step size/learning rate for the vertex positions")
    parser.add_argument('--lr_shader', type=float, default=1e-3, help="Step size/learning rate for the shader parameters")
    parser.add_argument('--upsample_iterations', type=int, nargs='+', default=[500, 1000, 1500], help="Iterations at which to perform mesh upsampling")
    parser.add_argument('--save_frequency', type=int, default=100, help="Frequency of mesh and shader saving (in iterations)")
    parser.add_argument('--visualization_frequency', type=int, default=100, help="Frequency of shader visualization (in iterations)")
    parser.add_argument('--visualization_views', type=int, nargs='+', default=[], help="Views to use for visualization. By default, a random view is selected each time")
    parser.add_argument('--device', type=int, default=0, choices=([-1] + list(range(torch.cuda.device_count()))), help="GPU to use; -1 is CPU")
    parser.add_argument('--weight_mask', type=float, default=2.0, help="Weight of the mask term")
    parser.add_argument('--weight_normal', type=float, default=0.1, help="Weight of the normal term")
    parser.add_argument('--weight_laplacian', type=float, default=40.0, help="Weight of the laplacian term")
    parser.add_argument('--weight_shading', type=float, default=1.0, help="Weight of the shading term")
    parser.add_argument('--weight_chamfer', type=float, default=1.0, help="Weight of the chamfer distance")
    parser.add_argument('--shading_percentage', type=float, default=0.75, help="Percentage of valid pixels considered in the shading loss (0-1)")
    parser.add_argument('--chamfer_percentage', type=float, default=0.75, help="Percentage of valid vertices considered in the chamfer loss (0-1)")
    parser.add_argument('--hidden_features_layers', type=int, default=3, help="Number of hidden layers in the positional feature part of the neural shader")
    parser.add_argument('--hidden_features_size', type=int, default=256, help="Width of the hidden layers in the neural shader")
    parser.add_argument('--fourier_features', type=str, default='positional', choices=(['none', 'gfft', 'positional','freenerf','cauchy']), help="Input encoding used in the neural shader")
    parser.add_argument('--activation', type=str, default='relu', choices=(['relu', 'sine']), help="Activation function used in the neural shader")
    parser.add_argument('--fft_scale', type=int, default=4, help="Scale parameter of frequency-based input encodings in the neural shader")

    #freenerf parameteres
    parser.add_argument('--L', type=int, default=16, help="hyperparameter that controls the maximum encoded frequency")
    parser.add_argument('--T', type=int, default=10000, help="the final iteration of frequency regularization")

    #multi-controlnetparameters
    parser.add_argument('--pipeline', type=str, default='our-mcnet', choices=(['our-mcnet','blackbox-mcnet']), help="Input encoding used in the neural shader")
    parser.add_argument("--cfg_text", default=7.5, type=int, help="hyperparameter that controls guidance scale")
    parser.add_argument("--prompt", required=False, type=str)
    parser.add_argument("--negative_prompt", required=False, type=str, default="")
    parser.add_argument('--controlnet', type=str, nargs='+', default=["ip2p", "normal"], help="names of controlnets, ip2p or normal")
    parser.add_argument('--conditioning_scale', type=int, nargs='+', default=[0.6, 0.4], help="condition array for controlnets")
    #parser.add_argument("--cfg_ip2p", default=1.0, type=float, help = "ip2p controlnet condition scale")
    #parser.add_argument("--cfg_normal", default=0.8, type=float, help = "normal controlnet condition scale")
    parser.add_argument("--seed", type=int, default=None) #23019
    parser.add_argument("--steps", default=10, type=int)
    parser.add_argument("--ip2p_use_full_precision", default=1, type=int)
    #latent scheduler
    parser.add_argument("--latent_step", default=0.0, type=float, help = "rendered_latent step")
    parser.add_argument("--latent_weight", default=0.9, type=float, help = "rendered_latent weight")
    parser.add_argument("--max_latent_weight", default=0.9, type=float, help = "max rendered_latent weight")
    parser.add_argument("--min_latent_weight", default=0.1, type=float, help = "min rendered_latent weight")
    parser.add_argument("--latent_change_interval", default=2000, type=int, help = "changing iteration for latent weight step")

    #misc
    parser.add_argument('--edit_begin', type=int, default=None, help="when to begin edit")
    parser.add_argument('--edit_frequency', type=int, default=10, help="editing frequency, when to edit")
    parser.add_argument('--edit_save_frequency', type=int, default=10, help="editing frequency, when to save edit")
    parser.add_argument('--comment', type=str, default="", help="")
   
    # Add module arguments
    EditViewSampler.add_arguments(parser) #TODO check this
    args = parser.parse_args()

    # Load configuration from YAML file if provided
    if args.config:
        with open(args.config, 'r') as config_file:
            yaml_config = yaml.safe_load(config_file)
        
        # Override argparse values with values from YAML file
        for key, value in yaml_config.items():
            setattr(args, key, value)

        args.output_dir, args.input_dir, args.experiment_name = Path(args.output_dir), Path(args.input_dir), Path(args.experiment_name)
        experiment_dir = args.output_dir /  args.experiment_name
        

    # Select the device
    device = torch.device('cpu')
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    print(f"Using device {device}")
    print(f"Training {args.experiment_name}")

    if args.pipeline == "our-mcnet":
        #Our pipeline
        print("using ours")
        ip2p_use_full_precision = False
        if args.ip2p_use_full_precision:
            print("using full precision")
            ip2p_use_full_precision = True
        else:
            print("using half precision")
        pipe = MultiControlNet(device, ip2p_use_full_precision=ip2p_use_full_precision, args=args) #TODO change naming pipe to model later
        update_view = our_update_view
    elif args.pipeline == "blackbox-mcnet":
        #Blackbox Multicontrolnet
        print("using blackbox")
        controlnet = [
            ControlNetModel.from_pretrained(CONTROLNET_IP2P_SOURCE, torch_dtype=torch.float16),
            ControlNetModel.from_pretrained(CONTROLNET_NORMAL_SOURCE, torch_dtype=torch.float16),
        ]

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            SD_SOURCE, controlnet=controlnet, torch_dtype=torch.float16, safety_checker = None
        )
        pipe.to(device)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        update_view = blackbox_update_view

        #pipe.enable_xformers_memory_efficient_attention()
        #pipe.enable_model_cpu_offload()
    elif args.pipeline =="controlnet-ip2p":
        #Our pipeline
        print("using ours")
        ip2p_use_full_precision = True
        pipe = ControlNet(device, ip2p_use_full_precision=ip2p_use_full_precision, args=args) #TODO change naming pipe to model later
        update_view = our_update_view

    args.output_dir, args.input_dir, args.experiment_name = Path(args.output_dir), Path(args.input_dir), Path(args.experiment_name)
    experiment_dir = args.output_dir /  args.experiment_name

    images_save_path = experiment_dir / "images"
    meshes_save_path = experiment_dir / "meshes"
    shaders_save_path = experiment_dir / "shaders"
    edits_save_path = experiment_dir / "edits"
    images_save_path.mkdir(parents=True, exist_ok=True)
    meshes_save_path.mkdir(parents=True, exist_ok=True)
    shaders_save_path.mkdir(parents=True, exist_ok=True)
    edits_save_path.mkdir(parents=True, exist_ok=True)
    
    if args.config:
        shutil.copy(args.config, os.path.join(experiment_dir, os.path.basename(args.config)))

    # Read the views one for ground truth, one for edited
    views = read_views(args.input_dir, scale=args.image_scale, device=device)
    edited_views = read_views(args.input_dir, scale=args.image_scale, device=device)
    
    # Obtain the initial mesh and compute its connectivity
    mesh_initial: Mesh = None
    if args.initial_mesh in mesh_generator_names:
        # Use args.initial_mesh as mesh generator name
        if args.input_bbox is None:
            raise RuntimeError("Generated meshes require a bounding box.")
        mesh_initial = generate_mesh(args.initial_mesh, edited_views, AABB.load(args.input_bbox), device=device)
    else:
        # Use args.initial_mesh as path to the mesh
        mesh_initial = read_mesh(args.initial_mesh, device=device)
    mesh_initial.compute_connectivity()

    # Load the bounding box or create it from the mesh vertices
    if args.input_bbox is not None: #TODO check if bbox are changed when resizing
        aabb = AABB.load(args.input_bbox)
    else:
        aabb = AABB(mesh_initial.vertices.cpu().numpy())
    aabb.save(experiment_dir / "bbox.txt")

    # Apply the normalizing affine transform, which maps the bounding box to 
    # a 2-cube centered at (0, 0, 0), to the views, the mesh, and the bounding box
    space_normalization = SpaceNormalization(aabb.corners)
    edited_views = space_normalization.normalize_views(edited_views)
    mesh_initial = space_normalization.normalize_mesh(mesh_initial)
    aabb = space_normalization.normalize_aabb(aabb)

    views = space_normalization.normalize_views(views)

    # Configure the renderer
    renderer = Renderer(device=device)
    renderer.set_near_far(edited_views, torch.from_numpy(aabb.corners).to(device), epsilon=0.5)#TODO optimize to edit views maybe?

    # Visualize the inputs before optimization
    visualize_views(views, show=False, save_path=experiment_dir / "views.png")
    visualize_mesh_as_overlay(renderer, views, mesh_initial, show=False, save_path=experiment_dir / "views_overlay.png")

    # Configure the edit view sampler
    view_sampler = EditViewSampler(views=edited_views, **EditViewSampler.get_parameters(args))
    
    # Create the optimizer for the vertex positions 
    # (we optimize offsets from the initial vertex position)
    lr_vertices = args.lr_vertices
    vertex_offsets = torch.zeros_like(mesh_initial.vertices) #TODO find a smart way to initialize vertices
    vertex_offsets.requires_grad = True
    optimizer_vertices = torch.optim.Adam([vertex_offsets], lr=lr_vertices)

    # Create the optimizer for the neural shader
    shader = NeuralShader(hidden_features_layers=args.hidden_features_layers,
                          hidden_features_size=args.hidden_features_size,
                          fourier_features=args.fourier_features,
                          activation=args.activation,
                          fft_scale=args.fft_scale,
                          last_activation=torch.nn.Sigmoid, 
                          device=device, L = args.L, T = args.T)
    optimizer_shader = torch.optim.Adam(shader.parameters(), lr=args.lr_shader)

    # Initialize the loss weights and losses
    loss_weights = {
        "mask": args.weight_mask,
        "normal": args.weight_normal,
        "laplacian": args.weight_laplacian,
        "shading": args.weight_shading,
        #"chamfer": 0.0
    }
    losses = {k: torch.tensor(0.0, device=device) for k in loss_weights}

    edit = False
    freenerf = False
    #noise = None
    #  # Create a tensor with the desired shape
    template_tensor = torch.empty((1, 4, 48, 64), dtype=torch.float16)

    # # Generate a random tensor with the same shape
    noise = torch.randn_like(template_tensor, device=device)

    adaptable = False
    changed_view  = 0
    edit_frequency = args.edit_frequency
    progress_bar = tqdm(range(1, args.iterations + 1))
    latent_weight = args.latent_weight
    refinement = False
    refinement_begin = [4000,8000,12000,16000]
    refinement_end = [6000,10000,14000,18000]
    for iteration in progress_bar:
        progress_bar.set_description(desc=f'Iteration {iteration}')

        # if iteration in refinement_begin:
        #     refinement = True
        # elif iteration in refinement_end:
        #     refinement = False
        #latent weight scheduling
        if args.latent_step and iteration > args.edit_begin and iteration % args.latent_change_interval == 0 and latent_weight < args.max_latent_weight and latent_weight>args.min_latent_weight: #fight darkness
            latent_weight += args.latent_step
            print("LATENT WEIGHT CHANGED TO: ", latent_weight)
        if iteration in args.upsample_iterations:
            # Upsample the mesh by remeshing the surface with half the average edge length
            e0, e1 = mesh.edges.unbind(1)
            average_edge_length = torch.linalg.norm(mesh.vertices[e0] - mesh.vertices[e1], dim=-1).mean()
            v_upsampled, f_upsampled = remesh_botsch(mesh.vertices.cpu().detach().numpy().astype(np.float64), mesh.indices.cpu().numpy().astype(np.int32), h=float(average_edge_length/2))
            v_upsampled = np.ascontiguousarray(v_upsampled)
            f_upsampled = np.ascontiguousarray(f_upsampled)

            mesh_initial = Mesh(v_upsampled, f_upsampled, device=device)
            mesh_initial.compute_connectivity()

            # Adjust weights and step size
            loss_weights['laplacian'] *= 4
            loss_weights['normal'] *= 4
            lr_vertices *= 0.75

            # Create a new optimizer for the vertex offsets
            vertex_offsets = torch.zeros_like(mesh_initial.vertices)
            vertex_offsets.requires_grad = True
            optimizer_vertices = torch.optim.Adam([vertex_offsets], lr=lr_vertices)
            # if iteration == args.upsample_iterations[len(args.upsample_iterations)-1]:
            #     update_gt_views(views, mesh_initial, renderer)
            #     shaded_path = images_save_path / "loss"
            #     shaded_path.mkdir(parents=True, exist_ok=True)
            #     plt.imsave(shaded_path / f'rendered_{iteration}_{0}.png', views[0].color.detach().cpu().numpy())
        
        # if iteration == 10000: #smurf-sonic-upscale2-changewithrendermask_any_lrmasklossincrease
        #     # Create a new optimizer for the vertex offsets
        #     loss_weights['mask'] *= 2
        #     lr_vertices *=2.0
        #     vertex_offsets = torch.zeros_like(mesh_initial.vertices)
        #     vertex_offsets.requires_grad = True
        #     optimizer_vertices = torch.optim.Adam([vertex_offsets], lr=lr_vertices)
        
        
        # if iteration == args.edit_begin:
        #     mesh_initial = mesh
        #     lr_vertices = args.lr_vertices
        #     # Create a new optimizer for the vertex offsets
        #     vertex_offsets = torch.zeros_like(mesh_initial.vertices)
        #     vertex_offsets.requires_grad = True
        #     optimizer_vertices = torch.optim.Adam([vertex_offsets], lr=lr_vertices)
        
        # if iteration == args.edit_begin+2000:
        #     mesh_initial = mesh
        #     lr_vertices *=0.75
        #     # Create a new optimizer for the vertex offsets
        #     vertex_offsets = torch.zeros_like(mesh_initial.vertices)
        #     vertex_offsets.requires_grad = True
        #     optimizer_vertices = torch.optim.Adam([vertex_offsets], lr=lr_vertices)

        # if iteration == args.edit_begin+4000:
        #     mesh_initial = mesh
        #     lr_vertices *=0.75
        #     # Create a new optimizer for the vertex offsets
        #     vertex_offsets = torch.zeros_like(mesh_initial.vertices)
        #     vertex_offsets.requires_grad = True
        #     optimizer_vertices = torch.optim.Adam([vertex_offsets], lr=lr_vertices)
        # if iteration == args.edit_begin:
        #     gt_vertices = mesh_initial.vertices
        #     loss_weights['chamfer'] = args.weight_chamfer
        # Deform the initial mesh
        mesh = mesh_initial.with_vertices(mesh_initial.vertices + vertex_offsets) #vertex_offsets.requires_grad = True
        views_subset, edited_subset, edit_view_indices = view_sampler(views, edited_views)
        
        # Perform antialiasing here because we cannot antialias after shading if we only shade a some of the pixels:
        gbuffers = renderer.render(edited_subset, mesh, channels=['mask', 'position', 'normal'], with_antialiasing=True)

        if loss_weights['mask'] > 0:
            losses['mask'] = mask_loss(edited_subset, gbuffers)
        if loss_weights['normal'] > 0:
            losses['normal'] = normal_consistency_loss(mesh)
        if loss_weights['laplacian'] > 0:
            losses['laplacian'] = laplacian_loss(mesh)
        if loss_weights['shading'] > 0:
            losses['shading'] = shading_loss(edited_subset, gbuffers, shader=shader, shading_percentage=args.shading_percentage, freenerf=freenerf, iteration = iteration, images_save_path = images_save_path, edit = edit)

        loss = torch.tensor(0., device=device)
        for k, v in losses.items():
            loss += v * loss_weights[k]

        # Optimize
        optimizer_vertices.zero_grad()
        optimizer_shader.zero_grad()
        loss.backward()
        optimizer_vertices.step()
        optimizer_shader.step()

        progress_bar.set_postfix({'loss': loss.detach().cpu()})

        rendered_images, normal_images, depth_images, masks = [], [], [], []
        view_indices = []
        # Visualizations
        if(args.visualization_frequency > 0) and shader and (iteration == 1 or iteration % args.visualization_frequency == 0 or iteration % 2004 == 0):
        #if (args.edit_begin) and (iteration >= args.edit_begin) and (iteration % args.edit_frequency == 0):
            import matplotlib.pyplot as plt
            with torch.no_grad():
                use_fixed_views = len(args.visualization_views) > 0
                view_indices = args.visualization_views if use_fixed_views else [np.random.choice(list(range(len(edited_subset))))]
                for vi in view_indices:
                    #debug_gt_view = views[vi] if use_fixed_views else views_subset[vi]
                    debug_view = edited_views[vi] if use_fixed_views else edited_subset[vi]
                    debug_gbuffer = renderer.render([debug_view], mesh, channels=['mask', 'position', 'normal','depth'], with_antialiasing=True)[0]
                    position, normal, depth = debug_gbuffer["position"], debug_gbuffer["normal"], debug_gbuffer["depth"]

                    view_direction = torch.nn.functional.normalize(debug_view.camera.center - position, dim=-1)

                    # Save the shaded rendering
                    shaded_image = shader(position, normal, view_direction, iteration) if freenerf else shader(position, normal, view_direction) 
                    shaded_image = shaded_image * debug_gbuffer["mask"] + (1-debug_gbuffer["mask"])
                    rendered_images.append(shaded_image)

                    
                    
                    shaded_path = (images_save_path / str(vi) / "shaded") if use_fixed_views else (images_save_path / "shaded")
                    shaded_path.mkdir(parents=True, exist_ok=True)
                    plt.imsave(shaded_path / f'neuralshading_{iteration}.png', shaded_image.cpu().numpy())

                    #depth_image = (depth - depth.min()) / (depth.max() - depth.min())

                    # depth_image = torch.cat([depth, depth, depth], dim=2)
                    # # Normalize the depth map between 0 and 1
                    # depth_min = depth_image.min()  # Minimum value in the tensor
                    # depth_max = depth_image.max()  # Maximum value in the tensor

                    # # Perform normalization
                    # depth_image = (depth_image - depth_min) / (depth_max - depth_min)  # Now [0, 1] range
                    # depth_image = depth_image * debug_gbuffer["mask"] +(1-debug_gbuffer["mask"])

                    # depth_image = 1 - depth_image
                    # depth_images.append(depth_image)

                    # depth_path = (images_save_path / str(vi) / "depth") if use_fixed_views else (images_save_path / "depth")
                    # depth_path.mkdir(parents=True, exist_ok=True)
                    # plt.imsave(depth_path / f'neuralshading_{iteration}.png', depth_image.cpu().numpy())

                    # mask = (debug_view.mask > 0) & (debug_gbuffer["mask"] > 0)
                    # #masks.append(mask)
                    # mask_path = (images_save_path / str(vi) / "mask") if use_fixed_views else (images_save_path / "mask")
                    # mask_path.mkdir(parents=True, exist_ok=True)
                    # plt.imsave(mask_path / f'neuralshading_{iteration}.png', mask.repeat(1, 1, 3).float().cpu().numpy())

                    # Save a normal map in camera space
                    normal_path = (images_save_path / str(vi) / "normal") if use_fixed_views else (images_save_path / "normal")
                    normal_path.mkdir(parents=True, exist_ok=True)
                    R = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], device=device, dtype=torch.float32)

                    #normal_image = normal @ debug_view.camera.R.T @ R.T 
                    normal_image = (0.5*(normal @ debug_view.camera.R.T @ R.T + 1)) * debug_gbuffer["mask"] + (1-debug_gbuffer["mask"])
                    
                    device = normal_image.device
                    normal_image = normal_image.cpu().numpy()
                    normal_image = np.array((normal_image - np.min(normal_image)) / (np.max(normal_image) - np.min(normal_image))) #fix floating point image rgb values issue
                    normal_images.append(torch.tensor(normal_image,device=device))
                    plt.imsave(normal_path / f'neuralshading_{iteration}.png', normal_image)
                                  
        #Edit views
        if (args.edit_begin) and (iteration >= args.edit_begin) and (iteration % edit_frequency == 0) and not refinement:       
            if rendered_images == []:
                with torch.no_grad():
                    use_fixed_views = len(args.visualization_views) > 0
                    view_indices = args.visualization_views if use_fixed_views else [np.random.choice(list(range(len(edited_subset))))]
                    for vi in view_indices:
                        #debug_gt_view = views[vi] if use_fixed_views else views_subset[vi]
                        debug_view = edited_views[vi] if use_fixed_views else edited_subset[vi]
                        debug_gbuffer = renderer.render([debug_view], mesh, channels=['mask', 'position', 'normal', 'depth'], with_antialiasing=True)[0]
                        position, normal, depth = debug_gbuffer["position"], debug_gbuffer["normal"], debug_gbuffer["depth"]
                        view_direction = torch.nn.functional.normalize(debug_view.camera.center - position, dim=-1)

                        # Save the shaded rendering
                        shaded_image = shader(position, normal, view_direction, iteration) if freenerf else shader(position, normal, view_direction) 
                        shaded_image = shaded_image * debug_gbuffer["mask"] + (1-debug_gbuffer["mask"])
                        rendered_images.append(shaded_image)
                        # mask = (debug_gt_view.mask > 0) & (debug_gbuffer["mask"] > 0)
                        # masks.append(mask)

                        # depth_image = torch.cat([depth, depth, depth], dim=2)
                        # # Normalize the depth map between 0 and 1
                        # depth_min = depth_image.min()  # Minimum value in the tensor
                        # depth_max = depth_image.max()  # Maximum value in the tensor

                        # # Perform normalization
                        # depth_image = (depth_image - depth_min) / (depth_max - depth_min)  # Now [0, 1] range
                        # depth_image = depth_image * debug_gbuffer["mask"] +(1-debug_gbuffer["mask"])

                        # depth_image = 1 - depth_image
                        # depth_images.append(depth_image)

                        R = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], device=device, dtype=torch.float32)
                        #normal_image = normal @ debug_view.camera.R.T @ R.T 
                        normal_image = (0.5*(normal @ debug_view.camera.R.T @ R.T + 1)) * debug_gbuffer["mask"] + (1-debug_gbuffer["mask"])
                        device = normal_image.device
                        normal_image = normal_image.cpu().numpy()
                        normal_image = np.array((normal_image - np.min(normal_image)) / (np.max(normal_image) - np.min(normal_image))) #fix floating point image rgb values issue
                        normal_images.append(torch.tensor(normal_image,device=device))
                        

            for ri, vi in enumerate(view_indices):
                gt_view = views[vi] if use_fixed_views else views_subset[vi]
                
                #mask = (gt_view.mask > 0) & (masks[ri] > 0)
                
                gt_color = gt_view.color #* 255.0
                rendered_image = rendered_images[ri] #* 255.0
                
                #edited_image = update_view(gt_color, rendered_image, model, args)
                normal_image = normal_images[ri]
                if depth_images!=[]:
                    depth_image = depth_images[ri]
                #mask = masks[ri]
                edited_image = update_view(gt_color, rendered_image, pipe, args, normal_image, iteration, images_save_path, noise=None, latent_weight=latent_weight, depth_image=None)
                
                #rgb_threshold = torch.tensor([0.95, 0.95, 0.95], device = rendered_image.device, dtype = rendered_image.dtype)
                #edited_mask = torch.any(edited_image <= rgb_threshold, dim=-1).unsqueeze(-1).float()
                # if iteration % args.visualization_frequency == 0:
                #     import matplotlib.pyplot as plt
                #     edit_mask_path = (images_save_path / str(vi) / "edit_mask") if use_fixed_views else (images_save_path / "edit_mask")
                #     edit_mask_path.mkdir(parents=True, exist_ok=True)
                #     plt.imsave(edit_mask_path / f'neuralshading_{iteration}.png', edited_mask.repeat(1, 1, 3).float().cpu().numpy())
                # rendered_image = Image.fromarray(rendered_image.type(torch.uint8).cpu().numpy())
                # gt_color = Image.fromarray(gt_color.type(torch.uint8).cpu().numpy())
                # rendered_image.save(shaded_path / f'rendered_{iteration}.png')
                # gt_color.save(shaded_path / f'gt_{iteration}.png')
                # exit()
                
                
                if use_fixed_views:
                    edited_views[vi].color = edited_image
                    #edited_views[vi].mask = edited_mask
                else:
                    edited_views[edit_view_indices[ri]].color = edited_image
                    #edited_views[edit_view_indices[ri]].mask = edited_mask
                
                # if iteration > 10151 and changed_view < 64:
                #     views[vi].color = rendered_image
                #     views[vi].mask = mask
                #     changed_view+=1
                #     if changed_view == 64:
                #         print("all views changed, switching to freenerf")
                #         if args.fourier_features == "freenerf":
                #             freenerf = True
               

        if (args.save_frequency > 0) and (iteration == 1 or iteration % args.save_frequency == 0):
            with torch.no_grad():
                mesh_for_writing = space_normalization.denormalize_mesh(mesh.detach().to('cpu'))
                write_mesh(meshes_save_path / f"mesh_{iteration:06d}.obj", mesh_for_writing)                                
            shader.save(shaders_save_path / f'shader_{iteration:06d}.pt')

    mesh_for_writing = space_normalization.denormalize_mesh(mesh.detach().to('cpu'))
    write_mesh(meshes_save_path / f"mesh_{args.iterations:06d}.obj", mesh_for_writing)

    if shader is not None:
        shader.save(shaders_save_path / f'shader_{args.iterations:06d}.pt')