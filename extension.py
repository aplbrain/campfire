import aws.sqs as sqs
from membrane_detection import membranes
import agents.scripts as scripts
import numpy as np
from cloudvolume import CloudVolume, exceptions
from agents import data_loader
from intern import array
import agents.sensor
import agents.scripts as scripts
from agents.swarm import Swarm
import time
import backoff
from scipy.ndimage.measurements import label

class Extension():
    def __init__(self, root_id, resolution, radius, unet_bound_mult, 
                 device, save, nucleus_id, time_point, endp, namespace):
        if type(root_id) == 'list':
            self.root_id = [int(r) for r in root_id]
        else:
            self.root_id = [int(root_id)]
        # self.public_root_id = scripts.get_public_seg_ids(int(root_id))
        # self.current_root_id = scripts.get_current_seg_ids(int(root_id))

        self.resolution = [int(x) for x in resolution]
        self.radius = [int(x) for x in radius]
        self.unet_bound_mult = float(unet_bound_mult)
        self.seg_root_id = -1
        self.device = device
        self.cnn_weights = 'thick'
        # self.cnn_weights = 'thin'
        self.save = save
        self.nucleus_id = nucleus_id
        self.time_point = time_point
        self.tic1 = time.time()
        self.namespace=namespace

    def get_bounds(self,endp):
        self.endpoint = np.divide(endp, self.resolution).astype('int')
        self.bound = get_bounds(self.endpoint, self.radius)
        self.bound_EM = get_bounds(self.endpoint, self.radius, self.unet_bound_mult)

    def gen_membranes(self, intern_pull=True, restitch_gaps=False, restitch_linear=False, conv_radius=3):
        tic = time.time()
        self.seg, self.em_big = self.get_data()
        toc=time.time()
        print("Seg EM Time", toc-tic)
        if type(self.seg) == str:
            if self.seg == "Out Of Bounds":
                return -1
            if np.sum(self.seg) == 0:
                return -2

        ## I will not stand integer overflow any longer ##
        ## Remaps from really large ints very fast ##
        self.seg_remap_dict = {i:k for i,k in enumerate(np.unique(self.seg))}
        from_values = np.array(list(self.seg_remap_dict.values()))
        to_values = np.array(list(self.seg_remap_dict.keys()))

        sort_idx = np.argsort(from_values)
        idx = np.searchsorted(from_values, self.seg,sorter = sort_idx)
        self.seg_remap = to_values[sort_idx][idx]

        self.mem_seg, self.seg, self.em, errors_gap, errors_linear = em_analysis(self.em_big, self.seg_remap, self.cnn_weights, self.device, self.bound_EM, intern_pull=intern_pull, restitch_gaps=restitch_gaps, restitch_linear=restitch_linear)
        self.errors = errors_gap + errors_linear
        self.mem_to_run = self.mem_seg#[int((unet_bound_mult-1)*radius[0]):int((unet_bound_mult+1)*radius[0]),
                   # int((unet_bound_mult-1)*radius[1]):int((unet_bound_mult+1)*radius[1]), :].astype(float)
        print("Restitch", restitch_gaps, restitch_linear)

        self.compute_vectors = scripts.precompute_membrane_vectors(self.mem_to_run, self.mem_to_run, conv_radius)

        l, n = label(self.errors)

        lab_counts = []
        for i in range(1,n+1):
            print(i)
            lab_counts.append(np.sum(l==i))
        if len(lab_counts) > 0:
            self.max_error = np.max(lab_counts)
            self.mean_errors = np.mean(lab_counts)
        else:
            self.max_error = 0
            self.mean_errors = 0

        print("Seg EM Restitch", time.time()-toc)

        return 1
    def run_agents(self,nsteps=200):
        tic = time.time()
        sensor_list = [
            (agents.sensor.BrownianMotionSensor(), .1),
            (agents.sensor.PrecomputedSensor(), .2),
            # (agents.sensor.MembraneSensor(self.mem_to_run), 0),
            ]
        self.agent_queue = self.create_queue(100, sampling_type="extension_ep")

        num_agents = len(self.agent_queue)
        # Spawn agents
        count = 1
        swarm = Swarm(
            data=self.compute_vectors,
            mem=self.mem_to_run,
            seg=self.seg,
            agent_queue=self.agent_queue,
            sensor_list=sensor_list,
            num_agents=num_agents,
            max_velocity=.9,
            parallel=False,
        )
        count = swarm.get_count()
        print(count, "agents spawned")
        print("\nAgent Spawning Prep Time", time.time() - tic)
        tic = time.time()
        for _ in range(nsteps):
            swarm.step()

        tic_step = time.time()
        steps_time = tic_step - tic

        print("Steps Time", steps_time, self.seg.dtype)
        # Save out data to file
        pos_histories = [a.get_position_history() for a in swarm.agents]
        tic = time.time()
        self.agent_merges(pos_histories)
        merge_time = time.time() - tic

        print("Merge Time", merge_time, self.seg.dtype)

    def save_agent_merges(self):
        duration = time.time()-self.tic1
        save_merges(self.save, self.merges, self.root_id[0], self.nucleus_id, self.time_point, self.endpoint,
                    self.weights_dict, self.bound, self.bound_EM, self.mem_seg, self.device, duration, self.n_errors, self.namespace)
   
    # Create a queue out of any of the above sampling methods
    def create_queue(self, n_pts, sampling_type="extension_only"):
        """
        Create spawn position list based on the desired sampling method

        Arguments:
            data_shape: (x,y,z) shape of the image / membrane data
            n_pts:  number of points per dimension to spawn linearly
            sampling type: "lin" or "synapse" to switch modes
            syns: for synapse spawning, the IDS to spawn at

        Returns:
            q: list of positions to spawn at
        """

        data_shape = self.em.shape

        if sampling_type == "extension":
            q = []
            ids, all_ids, centers, sizes_list = self.get_contacts(self.seg, self.root_id, conv_size=6)
            data_size = data_shape[0] * data_shape[1]
            ct=0
            ids = ids[ids != 0]
            # soma_table = scripts.get_soma(all_ids)
            polarity_table = scripts.is_dendrite(self.endpoint, all_ids)
            for i in ids:
                # if i != self.root_id and soma_table.get_soma(i) > 0:
                #     continue
                if i != self.root_id and not polarity_table.dendrite_filt(self.root_id, i):
                    continue
                centers_list = centers[i]
                sizes_ext = sizes_list[i]
                for j in range(len(centers_list)):
                    n_gen = int(sizes_ext[j] / data_size * n_pts)
                    if sizes_ext[j] > 0:
                        n_gen += 1

                    for _ in range(n_gen):
                        q.append([*centers_list[j]])
                    ct += 1
            # return q, soma_table, polarity_table
            return q, polarity_table
        if sampling_type == "extension_only":
            q = []
            centers_list, sizes_list = self.get_seg_slices()
            data_size = data_shape[0] * data_shape[1]
            ct=0
            for j in range(len(centers_list)):
                n_gen = int(sizes_list[j] / data_size * n_pts)
                if sizes_list[j] > 0:
                    n_gen += 1
                for _ in range(n_gen):
                    q.append([*centers_list[j]])
                ct += 1
        if sampling_type == "extension_ep":
            q = []
            center = data_shape[0]//2, data_shape[1]//2, data_shape[2]//2
            data_size = data_shape[0] * data_shape[1]
            for _ in range(n_pts):
                q.append([*center])
            return q

    def get_seg_slices(self):
        import cv2        
        import  copy
        seg_mask = copy.deepcopy(self.seg)
        # import pickle
        # pickle.dump(seg_mask, open("seg_mask.p", "wb"))
        print(self.seg_root_id, seg_mask[200,200,20], seg_mask.shape)

        seg_mask = np.in1d(seg_mask, self.seg_root_id).reshape(seg_mask.shape).astype(np.uint8)
        sizes_list = []
        centers_list = []
        for z in range(seg_mask.shape[2]//4, 3*seg_mask.shape[2]//4):
            if z < 2 or z > 198:
                continue
            seg_slice = seg_mask[:,:,z]
            if not np.any(seg_slice):
                continue

            _, x, y, _ = cv2.connectedComponentsWithStats(seg_slice)
            max_label = np.argmax(y[1:, 4])+1
            seg_slice = x == max_label
            points_slice = np.argwhere(seg_slice)

            if points_slice.shape[0] == 0:
                continue
            mean_point =  np.mean(points_slice, axis=0).astype(int)

            centers_list.append([*list(mean_point), int(z)])
            sizes_list.append(len(points_slice))
        return centers_list, sizes_list

    def agent_merges(self, pos_histories):
        tic = time.time()
        self.n_errors = np.sum(self.errors > 0)
        self.pos_matrix, self.merge_d = scripts.create_post_matrix(pos_histories, self.seg, self.mem_to_run.shape, merge=True)
        print("Merge_d time", time.time() - tic)
        self.merges = scripts.position_merge(self.endpoint, self.root_id, self.merge_d, self.endpoint*self.resolution, self.n_errors, self.mean_errors, self.max_error, self.seg_remap_dict)
        if self.merges.shape[0] > 0:
            #merges.to_csv(f"./merges_root_{root_id}_{endpoint}.csv")
            self.seg_ids = [str(x[:-1]) for x in self.merges.Extension]
            weights = [str(x) for x in self.merges.Weight]
            self.weights_dict = dict(zip(self.seg_ids, weights))
        else:
            self.weights_dict = {}
        return True
        
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def get_data(self, seg_or_sv = 'sv'):
        vol = CloudVolume("s3://bossdb-open-data/iarpa_microns/minnie/minnie65/em", use_https=True, mip=0)
        em = np.squeeze(vol[self.bound_EM[0]:self.bound_EM[1], self.bound_EM[2]:self.bound_EM[3], self.bound_EM[4]:self.bound_EM[5]])

        if seg_or_sv == 'seg':
            self.seg_root_id = self.public_root_id
            seg = np.squeeze(data_loader.get_seg(*self.bound_EM))
        elif seg_or_sv == 'sv':
            self.seg_root_id = self.root_id
            seg = data_loader.supervoxels(*self.bound_EM)
        # except (exceptions.OutOfBoundsError, exceptions.EmptyVolumeException):
        #     print("OOB")
        #     return "Out Of Bounds", "Out Of Bounds"
        return seg, em 

def get_bounds(endpoint, radius, mult=1):
    bound  = (endpoint[0] - int(mult*radius[0]), 
            endpoint[0] + int(mult*radius[0]),
            endpoint[1] - int(mult*radius[1]),
            endpoint[1] + int(mult*radius[1]),
            endpoint[2] - radius[2],
            endpoint[2] + radius[2])
    return bound


def get_endpoint(ep_param, delete, endp, root_id, nucleus_id, time_point):
    if ep_param == 'sqs':
        queue_url_endpts = sqs.get_or_create_queue('Endpoints_Test')
        ep_msg = sqs.get_job_from_queue(queue_url_endpts)
        root_id = int(ep_msg.message_attributes['root_id']['StringValue'])
        nucleus_id = int(ep_msg.message_attributes['nucleus_id']['StringValue'])
        time_point = int(ep_msg.message_attributes['time']['StringValue'])
        ep = [float(p) for p in ep_msg.body.split(',')]
        if delete:
            ep_msg.delete()
    else:
        root_id = root_id
        nucleus_id = nucleus_id
        time_point = time_point
        ep = [float(p) for p in endp]
    return root_id, nucleus_id, time_point, ep

def em_analysis(em, seg, cnn_weights, device, bound_EM, intern_pull=True, restitch_gaps=True, restitch_linear=True):

    # mem_seg = np.asarray(vol[bound_EM[0]:bound_EM[1], bound_EM[2]:bound_EM[3],bound_EM[4]:bound_EM[5]])
    # if np.sum(mem_seg) == 0:
    if intern_pull==False:
        if cnn_weights == 'thick':
            mem_seg = membranes.segment_membranes(em, pth="./membrane_detection/best_metric_model_segmentation2d_dict.pth", device_s=device)
        elif cnn_weights == 'thin':
            mem_seg = membranes.segment_membranes(em, pth="./membrane_detection/model_CREMI_2D.pth", device_s=device)
        # import pickle
        # pickle.dump(mem_seg, open(f"./data/reorient_mem_{bound_EM[0]}_{bound_EM[1]}_{bound_EM[2]}.p", "wb"))
        # from intern.remote.boss import BossRemote

        # rmt = BossRemote()

        # rmt.create_cutout_to_black(rmt.get_channel('membranes', 'microns', 'minnie65_8x8x40'),
        #                         0,
        #                         [bound_EM[0], bound_EM[1]],
        #                         [bound_EM[2], bound_EM[3]],
        #                         [bound_EM[4], bound_EM[5]])

        vol = array("bossdb://microns/minnie65_8x8x40/membranes")
        vol[bound_EM[4]:bound_EM[5],
                bound_EM[2]:bound_EM[3],
                bound_EM[0]:bound_EM[1]] = np.ascontiguousarray(np.rollaxis(mem_seg, 2, 0),dtype=np.uint64)
    else:
        vol = array("bossdb://microns/minnie65_8x8x40/membranes")
        mem_seg = vol[bound_EM[4]:bound_EM[5],
                bound_EM[2]:bound_EM[3],
                bound_EM[0]:bound_EM[1]] 
        mem_seg = np.moveaxis(np.array(mem_seg), 0, 2)
        print("Successful Intern Pull")
    print(restitch_gaps, restitch_linear)
    if restitch_gaps:
        errors_gap = scripts.detect_zero_locs(em, .25)
        errors_zero_template = scripts.detect_artifacts_template(em, errors_gap)
        warp_dict_reg, warp_dict_clean, warp_dict_flow = scripts.get_warp(em, errors_zero_template,
                                                                            patch_size=150,stride=75)
        em, seg, mem_seg = scripts.correct_with_flow(em, warp_dict_reg, alternate=True,stride=75, others_to_transform=[seg, mem_seg])
    
    else:
        errors_gap = np.zeros(em.shape[2])
        errors_zero_template = np.zeros(em.shape[2])

    if restitch_linear:
        em, seg, mem_seg = scripts.sofima_stitch(em, warp_scale=1.1, 
                                                   patch_size=150, stride=75, 
                                                   others_to_transform=[seg, mem_seg])
        mem_seg[512] = 1
        mem_seg[:, 512] =1

    return mem_seg, seg, em, errors_gap, errors_zero_template

def save_merges(save, merges, root_id, nucleus_id, time_point, endpoint, weights_dict, bound,
                bound_EM, mem_seg, device, duration, n_errors, namespace):
    if type(root_id) == list:
        root_id = root_id[0]
    if save == "pd":
        print("Save", f"./data/merges_{root_id}_{endpoint}.csv")
        merges.to_csv(f"./data/merges_{root_id}_{endpoint}.csv")
    if save == "nvq":
        import neuvueclient as Client
        print("DURATION", duration, root_id, endpoint, "\n")
        metadata = {'time':str(time_point), 
                    'duration':str(duration), 
                    'device':device,
                    'bbox':[str(x) for x in bound], 
                    'bbox_em':[str(x) for x in bound_EM],
                    'n_errors':str(n_errors),
                    }
        C = Client.NeuvueQueue("https://queue.neuvue.io")
        end = [int(x) for x in endpoint]
        if type(root_id) == int:
            root_id = str(root_id)
        elif type(root_id) == list or type(root_id) == np.ndarray:
            root_id = str(root_id[0])
        print("Posted", str(root_id), str(nucleus_id), end, weights_dict, metadata)
        C.post_agent(str(root_id), str(nucleus_id), end, weights_dict, metadata) 
        # pickle.dump(mem_seg, open(f"./data/INTERNmem_{duration}_{root_id}_{endpoint}.p", "wb"))

        # pickle.dump(bound_EM, open(f"./data/INTERNBOUNDS_{duration}_{root_id}_{endpoint}.p", "wb"))

def visualize_merges(merges_root, seg, root_id):
    mset = set(merges_root.M1)
    mset.update(set(merges_root.M2))
    sum_weight = np.sum(np.asarray(merges_root["Weight"]))
    seg = np.asarray(seg)
    opacity_merges = np.zeros_like(seg).astype(float)
    seg_merge_output = np.zeros_like(seg).astype(float)
    for i, seg_id in enumerate(mset):
        if float(seg_id) == float(root_id):
            opacity_merges[seg == seg_id] = 1
            seg_merge_output[seg == seg_id] = -1
            continue

        opacity_merges[seg == seg_id] = merges_root[(merges_root.M1 == seg_id) | (merges_root.M2 == seg_id)]["Weight"] / sum_weight
        seg_merge_output[seg == seg_id] = i
    seg_merge_output[seg_merge_output==0]=np.nan
    return opacity_merges, seg_merge_output
