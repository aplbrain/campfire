import aws.sqs as sqs
from membrane_detection import membranes
import agents.scripts as scripts
import numpy as np
from cloudvolume import CloudVolume
from cloudvolume import exceptions as cvExceptions
from agents import data_loader
from caveclient import CAVEclient
from intern import array
import agents.sensor
import agents.scripts as scripts
from agents.swarm import Swarm
import time
client = CAVEclient('minnie65_phase3_v1')

class Extension():
    def __init__(self, root_id, resolution, radius, unet_bound_mult, 
                 device, save, nucleus_id, time_point, endp):
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

    def get_bounds(self,endp):
        self.endpoint = np.divide(endp, self.resolution).astype('int')
        self.bound = get_bounds(self.endpoint, self.radius)
        self.bound_EM = get_bounds(self.endpoint, self.radius, self.unet_bound_mult)

    def gen_membranes(self):

        self.seg, self.em = self.get_data()
        self.mem_to_run, self.mem_seg, self.error_dict, self.compute_vectors = em_analysis(self.em, self.cnn_weights, self.unet_bound_mult, self.radius, self.device, self.bound_EM)
    
    def membrane_seg_save(self):

        self.seg, self.em = self.get_data(seg_or_sv='membrane')
        if type(self.seg) == int:
            return False

        self.success = em_analysis(self.em, self.cnn_weights, self.unet_bound_mult, self.radius, self.device, self.bound_EM)
        return True

    def run_agents(self):
        tic = time.time()
        sensor_list = [
            (agents.sensor.BrownianMotionSensor(), .1),
            (agents.sensor.PrecomputedSensor(), .2),
            (agents.sensor.MembraneSensor(self.mem_to_run), 0),
            ]
        agent_queue = self.create_queue(250, sampling_type="extension_only")

        num_agents = len(agent_queue)
        # Spawn agents
        count = 1
        swarm = Swarm(
            data=self.compute_vectors,
            mem=self.mem_to_run,
            seg=self.seg,
            agent_queue=agent_queue,
            sensor_list=sensor_list,
            num_agents=num_agents,
            max_velocity=.9,
            parallel=False,
        )
        count = swarm.get_count()
        print(count, "agents spawned")
        print("\nAgent Spawning Prep Time", time.time() - tic)
        tic = time.time()
        for _ in range(300):
            swarm.step()

        tic_step = time.time()
        steps_time = tic_step - tic

        print("Steps Time", steps_time)
        # Save out data to file
        pos_histories = [a.get_position_history() for a in swarm.agents]
        tic = time.time()

        self.agent_merges(pos_histories)
        merge_time = time.time() - tic

        print("Merge Time", merge_time)

    def save_agent_merges(self):
        duration = time.time()-self.tic1
        save_merges(self.save, self.merges, self.root_id[0], self.nucleus_id, self.time_point, self.endpoint,
                    self.weights_dict, self.bound, self.bound_EM, self.mem_seg, self.device, duration, self.error_dict)
   
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
            soma_table = scripts.get_soma(all_ids)
            polarity_table = scripts.is_dendrite(self.endpoint, all_ids)
            for i in ids:
                if i != self.root_id and soma_table.get_soma(i) > 0:
                    continue
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
            return q, soma_table, polarity_table
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
            return q

    def get_seg_slices(self):
        import  copy
        seg_mask = copy.deepcopy(self.seg)
        import pickle
        pickle.dump(seg_mask, open("seg_mask.p", "wb"))
        seg_mask = np.in1d(seg_mask, self.seg_root_id).reshape(seg_mask.shape)

        sizes_list = []

        points = np.argwhere(seg_mask)
        points_z = points[:,2]
        centers_list = []
        for z in range(np.min(points_z), np.max(points_z)+1):
            if z < 2 or z > 198:
                continue
            points_slice = points[points_z == z]
            # seg_slice = seg[:,:,z]
            # seg_mask_slice = np.in1d(seg_slice, root_id).reshape(seg_slice.shape)
            # points_slice = np.argwhere(seg_mask_slice)
            centers_list.append([*list(np.mean(points_slice, axis=0).astype(int))])
            sizes_list.append(len(points_slice))
        return centers_list, sizes_list

    def agent_merges(self, pos_histories):

        self.pos_matrix, self.merge_d = scripts.create_post_matrix(pos_histories, self.seg, self.mem_to_run.shape, merge=True)
        self.merges = scripts.position_merge(self.endpoint, self.root_id, self.merge_d,  self.endpoint*self.resolution, self.error_dict, self.seg)

    #     merges = scripts.merge_paths(pos_histories,seg_ids, ep, root_id, soma, polarities)
        if self.merges.shape[0] > 0:
            #merges.to_csv(f"./merges_root_{root_id}_{endpoint}.csv")
            self.seg_ids = [int (x) for x in self.merges.Extension]
            weights = [int(x) for x in self.merges.Weight]
            self.weights_dict = dict(zip(self.seg_ids, weights))
        else:
            self.weights_dict = {}
        return True
    def get_data(self, seg_or_sv = 'sv'):
        
        vol = CloudVolume("s3://bossdb-open-data/iarpa_microns/minnie/minnie65/em", use_https=True, mip=0)

        try:
            em = np.squeeze(vol[self.bound_EM[0]:self.bound_EM[1], self.bound_EM[2]:self.bound_EM[3], self.bound_EM[4]:self.bound_EM[5]])

            if seg_or_sv == 'seg':
                self.seg_root_id = self.public_root_id
                seg = np.squeeze(data_loader.get_seg(*self.bound))
            elif seg_or_sv == 'sv':
                self.seg_root_id = self.root_id
                seg = data_loader.supervoxels(*self.bound)
            elif seg_or_sv == 'membranes':
                seg = 0
        except (cvExceptions.OutOfBoundsError, cvExceptions.EmptyVolumeException):
            return -1, -1

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


def em_analysis(em, cnn_weights, unet_bound_mult, radius, device, bound_EM):

    # errors_shift, errors_zero, error_dict = scripts.shift_detect(em, 100, 1, 1, 10)
    # vol = array("bossdb://microns/minnie65_8x8x40/membranes", axis_order="XYZ")
    # mem_seg = np.asarray(vol[bound_EM[0]:bound_EM[1], bound_EM[2]:bound_EM[3],bound_EM[4]:bound_EM[5]])
    # if np.sum(mem_seg) == 0:
    if 0 == 0:
        if cnn_weights == 'thick':
            mem_seg = membranes.segment_membranes(em, pth="./membrane_detection/best_metric_model_segmentation2d_dict.pth", device_s=device)
        elif cnn_weights == 'thin':
            mem_seg = membranes.segment_membranes(em, pth="./membrane_detection/model_CREMI_2D.pth", device_s=device)

    # mem_seg = scripts.remove_shifts(mem_seg, errors_shift+errors_zero, zero_or_remove='zero')

    # mem_to_run = mem_seg[int((unet_bound_mult-1)*radius[0]):int((unet_bound_mult+1)*radius[0]),
    #                 int((unet_bound_mult-1)*radius[1]):int((unet_bound_mult+1)*radius[1]), :].astype(float)
    # compute_vectors = scripts.precompute_membrane_vectors(mem_to_run, mem_to_run, 3)
    from intern.remote.boss import BossRemote

    rmt = BossRemote()
    rmt.create_cutout_to_black(rmt.get_channel('membranes', 'microns', 'minnie65_8x8x40'),
                               0,
                               [bound_EM[0], bound_EM[1]],
                               [bound_EM[2], bound_EM[3]],
                               [bound_EM[4], bound_EM[5]])

    vol = array("bossdb://microns/minnie65_8x8x40/membranes")

    vol[bound_EM[4]:bound_EM[5],
        bound_EM[2]:bound_EM[3],
        bound_EM[0]:bound_EM[1]] = np.zeros((mem_seg.shape[2], mem_seg.shape[1], mem_seg.shape[0]))
    print("1")
    vol[bound_EM[4]:bound_EM[5],
        bound_EM[2]:bound_EM[3],
        bound_EM[0]:bound_EM[1]] = np.ascontiguousarray(np.rollaxis(mem_seg, 2, 0),dtype=np.uint64)    # import pickle
    # pickle.dump(mem_seg, open("ex_seg.p", "wb"))
    # vol[bound_EM[0]:bound_EM[1], bound_EM[2]:bound_EM[3],bound_EM[4]:bound_EM[5]] = mem_seg.astype(np.uint64)

    return True

def save_merges(save, merges, root_id, nucleus_id, time_point, endpoint, weights_dict, bound,
                bound_EM, mem_seg, device, duration, error_dict):
    if type(root_id) == list:
        root_id = root_id[0]
    if save == "pd":
        merges.to_csv(f"./data/merges_{root_id}_{endpoint}.csv")
    if save == "nvq":
        import neuvueclient as Client
        print("DURATION", duration, root_id, endpoint, "\n")
        metadata = {'time':str(time_point), 
                    'duration':str(duration), 
                    'device':device,
                    'bbox':[str(x) for x in bound], 
                    'bbox_em':[str(x) for x in bound_EM],
                    'error_dict':error_dict,
                    }
        C = Client.NeuvueQueue("https://queue.neuvue.io")
        end = [str(x) for x in endpoint]
        
        print(str(root_id[0]), str(nucleus_id), end, weights_dict, metadata)
        C.post_agent(str(root_id[0]), str(nucleus_id), end, weights_dict, metadata)
        
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
