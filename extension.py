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
from scipy.ndimage import label

class Extension():
    def __init__(self, root_id, resolution, radius, unet_bound_mult, 
                 device, save, nucleus_id, time_point, namespace, direction_test, point_id):
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
        self.point_id=point_id
        # self.cnn_weights = 'thin'
        self.save = save
        self.nucleus_id = nucleus_id
        self.time_point = time_point
        self.tic1 = time.time()
        self.namespace=namespace
        self.direction_test = direction_test

    def get_bounds(self,endp):
        self.endpoint = np.divide(endp, self.resolution).astype('int')
        self.bound = get_bounds(self.endpoint, self.radius)
        # z_mult = 2 if self.direction_test else 1
        self.bound_EM = get_bounds(self.endpoint, self.radius, self.unet_bound_mult)

    def gen_membranes(self, intern_pull=True, restitch_gaps=False):
        ex_slices = 4

        tic = time.time()
        self.seg, self.em_big = self.get_data()
        if self.seg is None:
            print("EMPTY VOLUME")
            return -3
        for i in range(1, self.seg.shape[2]):
            if np.sum((self.seg[..., i-1] - self.seg[..., i]) != 0) < 1000:
                self.seg[..., i-1] = 0
        toc=time.time()
        print("Seg EM Time", toc-tic)
        if type(self.seg) == str:
            if self.seg == "Out Of Bounds":
                self.merges={}
                return -1
            if np.sum(self.seg) == 0:
                self.merges={}
                return -2
        if self.direction_test:
            self.middle_spot = self.seg.shape[2] // 2
            above = np.sum(self.seg[:, :, self.middle_spot:] == self.root_id)
            below = np.sum(self.seg[:, :, :self.middle_spot] == self.root_id)
            if above > below:
                self.direction = -1
                self.seg = self.seg[:, :, :self.middle_spot+ex_slices]
                self.em_big = self.em_big[:, :, :self.middle_spot+ex_slices]
                self.bound_EM[5] -= self.em_big.shape[2] - 2*ex_slices
            elif below > above:
                self.direction = 1
                self.seg = self.seg[:, :, self.middle_spot - ex_slices:]
                self.em_big = self.em_big[:, :,self.middle_spot - ex_slices:]
                self.bound_EM[4] += self.em_big.shape[2] - 2*ex_slices
            elif below == above and below > 0:
                print("edge case!!")
                self.direction = 0
                self.direction_test = False
            else:
                print("ROOT ID NOT FOUND, CRASHING")
                return 0
        else:
            self.direction = 0

        ## I will not stand integer overflow any longer ##
        ## Remaps from really large ints very fast ##
        unique_seg = np.unique(self.seg)
        self.seg_remap_dict = {i:k for i,k in enumerate(unique_seg)}
        self.seg_remap_dict_r = {k:i for i,k in enumerate(unique_seg)}
        from_values = np.array(list(self.seg_remap_dict.values()))
        to_values = np.array(list(self.seg_remap_dict.keys()))

        sort_idx = np.argsort(from_values)
        idx = np.searchsorted(from_values, self.seg,sorter = sort_idx)
        self.seg_remap = to_values[sort_idx][idx]

        self.mem_seg, self.seg, self.em, errors_gap, errors_linear = em_analysis(self.em_big, self.seg_remap, self.cnn_weights, self.device, self.bound_EM, intern_pull=intern_pull, restitch_gaps=restitch_gaps, direction=self.direction)
        self.errors = errors_gap + errors_linear
        self.mem_to_run = self.mem_seg#[int((unet_bound_mult-1)*radius[0]):int((unet_bound_mult+1)*radius[0]),
                   # int((unet_bound_mult-1)*radius[1]):int((unet_bound_mult+1)*radius[1]), :].astype(float)

        # self.seg = self.seg[:512,:512]
        # self.em = self.em[:512,:512]
        # self.mem_seg = self.mem_seg[:512,:512]
        print("Restitch", restitch_gaps)

        # self.compute_vectors = scripts.precompute_membrane_vectors(self.mem_to_run, self.mem_to_run, conv_radius)

        l, n = label(self.errors)

        lab_counts = []
        for i in range(1,n+1):
            lab_counts.append(np.sum(l==i))
        if len(lab_counts) > 0:
            self.max_error = np.max(lab_counts)
            self.mean_errors = np.mean(lab_counts)
        else:
            self.max_error = 0
            self.mean_errors = 0

        print("Seg EM Restitch", time.time()-toc)

        return 1

    def dist_transform_merge(self):
        import scipy
        import cc3d
        ms = 1-scipy.ndimage.binary_dilation(self.mem_seg)

        skel_ret = np.zeros(ms.shape)

        for i in range(1, ms.shape[2]-1):
            print(i, np.sum(ms[..., i]>0))

            if np.sum(ms[..., i] > 0) < 1000:
                skel_ret[..., i] = 0
            skel_ret[..., i] = scipy.ndimage.distance_transform_edt(ms[:, :, i])
            
        morph = scipy.ndimage.morphological_laplace(skel_ret, (50, 50, 25))
        l = scipy.ndimage.minimum_filter(morph, (5, 5, 3))

        skel_ret= l < l.max()/8

        seg_rid = self.seg_remap_dict_r[int(self.root_id[0])]
        seg_bin = self.seg == seg_rid
        seg_bin = cc3d.largest_k(
        seg_bin, k=5, 
        connectivity=26, delta=0,
        )
        seg_middle = seg_bin[self.radius[0], self.radius[1]]
        seg_middle = seg_middle[seg_middle != 0]
        bc_seg = np.bincount(seg_middle)
        if len(bc_seg) == 0:
            print("Major Shift")
            self.merge_d = {}
            return
        seg_bin[seg_bin != np.argmax(bc_seg)] = 0
        seg_bin = seg_bin > 0

        structure = np.zeros((3,3,3))
        structure[:, :, 1] = 1
        structure[1, 1, :] = 1


        edt_labelled = scipy.ndimage.label(skel_ret, structure=structure)[0]
        edt_masked = edt_labelled.copy()
        edt_masked[~seg_bin] = 0
        counts = np.bincount(edt_masked.flatten())

        map_dict = {i[0]: counts[i[0]] for i in np.argwhere(counts > 0)}
        results_dict = {}
        results_thresh = {}

        seg_pix = np.zeros_like(edt_labelled)
        for k, v in map_dict.items():
            if k == 0:
                continue
            seg_pix[edt_labelled == k] = k
            bins = np.bincount(self.seg[edt_labelled == k])
            bins_arg = np.argwhere(bins)
            for b in bins_arg:
                b = int(np.squeeze(b))
                to_merge = b
                if bins[b] < 10 or to_merge == 0 or to_merge == seg_rid:
                    continue
                try:
                    results_dict[(seg_rid, to_merge)] += bins[b]*v

                except:
                    results_dict[(seg_rid, to_merge)] = bins[b]*v
                    
        seg_direcs = np.argwhere(np.sum(np.sum(self.seg_remap == seg_rid, axis=0), axis=0) > 0)

        res_list  = list(results_dict.values())
        if len(res_list) > 0:
            max_val = np.max(res_list)
            for k, v in results_dict.items():
                if v > .05*max_val:
                    ext_direcs = np.argwhere(np.sum(np.sum(self.seg_remap == k[1], axis=0), axis=0) > 0)
                #         print(k, ext_direcs)
                    if self.direction == 1:
                        if np.max(seg_direcs) > np.min(ext_direcs):
                            continue
                    if self.direction == -1:
                        if np.min(seg_direcs) <= np.min(ext_direcs):
                            continue
                    results_thresh[k] = v / max_val
        else:
            results_thresh = {}
        self.merge_d = results_thresh

    def run_agents(self,nsteps=200, n_agents=100):
        tic = time.time()
        sensor_list = [
            (agents.sensor.BrownianMotionSensor(), .1),
            (agents.sensor.PrecomputedSensor(), .2),
            (agents.sensor.MembraneSensor(self.mem_to_run), 0),
            ]
        self.agent_queue = self.create_queue(n_agents, sampling_type="extension_ep")
        print("queue", self.agent_queue)
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

    def save_agent_merges(self, error=False):
        if error:
            import neuvueclient as Client
            C = Client.NeuvueQueue("https://queue.neuvue.io")
            C.patch_point(self.point_id, agents_status='extension_completed')
            return
        duration = time.time()-self.tic1
        save_merges(self.save, self.merges, self.root_id[0], self.nucleus_id, self.time_point, self.endpoint,
                    self.weights_dict, self.bound, self.bound_EM, self.mem_seg, self.device, duration, self.n_errors, self.namespace, self.point_id)
   
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
            if self.direction_test:
                if self.direction == -1:
                    center = data_shape[0]//2, data_shape[1]//2, data_shape[2]-1
                elif self.direction == 1:
                    center = data_shape[0]//2, data_shape[1]//2, 1
            else:
                center = data_shape[0]//2, data_shape[1]//2, data_shape[2]//2
            print("CENTER", self.direction, center)
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
    
    def distance_merge_save(self):
        self.n_errors = np.sum(self.errors > 0)
        self.merges = scripts.position_merge(self.endpoint, self.root_id, self.merge_d, self.endpoint*self.resolution, self.n_errors, self.mean_errors, self.max_error, self.seg_remap_dict)
        if self.merges.shape[0] > 0:
            #merges.to_csv(f"./merges_root_{root_id}_{endpoint}.csv")
            self.seg_ids = [str(x[:-1]) for x in self.merges.Extension]
            weights = [str(x) for x in self.merges.Weight]
            self.weights_dict = dict(zip(self.seg_ids, weights))
        else:
            self.weights_dict = {}
            
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def get_data(self, seg_or_sv = 'sv'):
        try:
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
        except:
            return None, None

def get_bounds(endpoint, radius, mult=1, z_mult=1):

    bound  = [endpoint[0] - int(mult*radius[0]), 
            endpoint[0] + int(mult*radius[0]),
            endpoint[1] - int(mult*radius[1]),
            endpoint[1] + int(mult*radius[1]),
            endpoint[2] -  int(z_mult*radius[2]),
            endpoint[2] + int(z_mult*radius[2])]
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

def em_analysis(em, seg, cnn_weights, device, bound_EM, intern_pull=True, restitch_gaps=True, direction=0, zero_threshold=.5):

    # mem_seg = np.asarray(vol[bound_EM[0]:bound_EM[1], bound_EM[2]:bound_EM[3],bound_EM[4]:bound_EM[5]])
    # if np.sum(mem_seg) == 0:
    print("PULL", intern_pull)
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

    if restitch_gaps:
        errors_gap = scripts.detect_zero_locs(em, .25)
        errors_zero_template = scripts.detect_artifacts_template(em, errors_gap)

        first_slide = np.min(np.argwhere(errors_gap==0))
        last_slide = np.max(np.argwhere(errors_gap==0))
        em = em[..., first_slide:last_slide+1]
        mem_seg = mem_seg[..., first_slide:last_slide+1]
        seg = seg[..., first_slide:last_slide+1]

        em, mem_seg, seg = scripts.combined_stitch(em, errors_zero_template, direction,
                                                    patch_size = 180, stride = 45, alternate=True,
                                                    others_to_transform=[mem_seg, seg])
        em = em[:560,:560]
        mem_seg = mem_seg[:560,:560]
        seg = seg[:560,:560]
    else:
        errors_gap = np.zeros(em.shape[2])
        errors_zero_template = np.zeros(em.shape[2])


    start=0
    while True:
        last_frame = em[:,:,start]
        print(1, start, np.sum(last_frame == 0) / np.product(mem_seg.shape[:2]))
        if np.sum(last_frame == 0) / np.product(mem_seg.shape[:2]) > zero_threshold:
            mem_seg[:,:,start] = np.ones(mem_seg.shape[:2])
        else:
            break
        start+=1
    start=-1
    while True:
        print(start)
        last_frame = em[:,:,start]
        if np.sum(last_frame == 0) / np.product(mem_seg.shape[:2]) > zero_threshold:
            mem_seg[:,:,start] = np.ones(mem_seg.shape[:2])
        else:
            break
        start-=1
    return mem_seg, seg, em, errors_gap, errors_zero_template

def save_merges(save, merges, root_id, nucleus_id, time_point, endpoint, weights_dict, bound,
                bound_EM, mem_seg, device, duration, n_errors, namespace, points_id):
    if type(root_id) == list:
        root_id = root_id[0]
    if save == "pd":
        print("Save", f"./data/merges_{root_id}_{endpoint}.csv")
        merges.to_csv(f"./data/merges_{root_id}_{endpoint}.csv")
    if save == "nvq":
        import neuvueclient as Client
        print("DURATION", duration, root_id, endpoint, "\n")
        error=False
        if error:
            metadata={'oob':True}
        else:
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
        if len(merges) == 0:
            weights_dict={}
        C.post_agent(str(root_id), str(nucleus_id), end, weights_dict, metadata, namespace) 
        C.patch_point(points_id, agents_status='extension_completed')
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
