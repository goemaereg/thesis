from .random_simple_generator import SimpleRandomFetcher
import numpy as np
import torch.utils.data as data


#######################################################################
#                            - Example 1 -
# Consider the following as an example of how our data generation
#   objects are implemented.
#######################################################################

class TenClassesRandomFetcher(SimpleRandomFetcher):
    def __init__(self, n_instances=1, random_n_instances=False, type_noise=False, overlapping=False):
        super(TenClassesRandomFetcher, self).__init__()
        self.n_instances=n_instances
        self.random_n_instances = random_n_instances
        self.type_noise = type_noise

    # def setup0001(self, general_meta_setting, explanation_setting, s=None):
    #     self.s = s if s is not None else (512,512) # image shape

    def uniform_random_draw(self):
        types = 10 if self.type_noise else 9
        y0 = np.random.randint(types)
        bg_rand = np.random.randint(3)
        return self.draw_n_instances(y0, bg_rand, self.n_instances)

    def draw_one_sample_no_background(self, y0):
        # y0 is 0,1,...,9
        cobj, cimg, heatmap, variables = None, None, None, None
        if y0 == 0:
            cobj, cimg, heatmap, variables = self.get_random_CCellX()
        elif y0 == 1:
            cobj, cimg, heatmap, variables = self.get_random_CCellMX()
        elif y0 == 2:
            cobj, cimg, heatmap, variables = self.get_random_CCellPX()
        elif y0 == 3:
            cobj, cimg, heatmap, variables = self.get_random_RCellX()  # red
        elif y0 == 4:
            cobj, cimg, heatmap, variables = self.get_random_RCellXB()  # green
        elif y0 == 5:
            cobj, cimg, heatmap, variables = self.get_random_RCellXC()  # blue
        elif y0 == 6:
            tfraction = 0.
            while tfraction < 0.3:
                cobj, cimg, heatmap, variables = self.get_random_CCellTX()
                tailpos = cobj.parts['tailpos'].reshape(-1)
                tfraction = np.round(np.sum(tailpos) / len(tailpos) * 100., 2)
            #     if tfraction < 0.3: print('tfraction (reject)',tfraction)
            # print('tfraction',tfraction)
        elif y0 == 7:
            cobj, cimg, heatmap, variables = self.get_random_CCellTX3()
        elif y0 == 8:
            cobj, cimg, heatmap, variables = self.get_random_CCellTX8()
        elif y0 == 9:
            cobj, cimg, heatmap, variables = None, np.zeros(self.s + (3,)), np.zeros(self.s[:2]), {'type': 'NOISE'}
        # print('[%s] cimg.shape:%s, heatmap.shape:%s'%(str(y0), str(cimg.shape),str(heatmap.shape)))

        return cobj, cimg, heatmap, variables


    def draw_one_sample(self, y0, bg_rand):
        # y0 is 0,1,...,9
        # bg_rand is 0, 1, 2
        cobj, cimg, heatmap, variables = self.draw_one_sample_no_background(y0)
                
        self.background_setting['type'] = bg_rand
        bg = self.generate_background()
        if bg is not None:
            ep = 1e-2
            pos = np.stack(((cimg[:,:,0]<ep),(cimg[:,:,1]<ep),(cimg[:,:,2]<ep))).transpose((1,2,0))
            cimg =  cimg + pos * bg 
        cimg = np.clip(cimg, a_min=0., a_max=1.)

        variables['y0'] = y0
        variables['bg_rand'] = bg_rand

        return cobj, cimg, heatmap, variables

    def draw_n_instances(self, y0, bg_rand, n_instances=1):
        image_list = []
        heatmap_list = []
        variable_list = []
        if self.random_n_instances:
            n_instances = np.random.randint(1, n_instances + 1)

        for _ in range(n_instances):
            # cimg: (H,W,C) ; heatmap: (H,w), variables: dict
            _, cimg, heatmap, variables = self.draw_one_sample_no_background(y0)
            image_list.append(cimg)
            heatmap_list.append(heatmap)
            variable_list.append(variables)

        # generate background image
        self.background_setting['type'] = bg_rand
        # bg: (H, W, 3)
        bg = self.generate_background()

        # stack instances as layers
        image_list.append(bg)
        # cimg: (n_intances + 1, H, W, 3)
        cimg = np.stack(image_list, axis=0)
        # heatmap: (n_instances, H, W)
        heatmap = np.stack(heatmap_list, axis=0)

        variables = variable_list[0]
        variables['y0'] = y0
        variables['bg_rand'] = bg_rand

        return None, cimg, heatmap, variables


class TenClassesPyIO(data.Dataset, TenClassesRandomFetcher):
    def __init__(self, n_instances=1, random_n_instances=False, type_noise=False, overlapping=False):
        super(TenClassesPyIO, self).__init__(n_instances=n_instances, random_n_instances=random_n_instances,
                                             type_noise=type_noise, overlapping=overlapping)
        self.x, self.y = [], []

    def __getitem__(self, index):
        return np.array(self.x[index]), np.array(self.y[index])

    def __len__(self):
        return self.data_size

    def setup_training_0001(self, general_meta_setting=None, explanation_setting=None, data_size=12, 
        realtime_update=False):
        self.x, self.y = [], []
        self.data_size = data_size

        self.setup0001(general_meta_setting, explanation_setting, s=None)
        for i in range(data_size):
            if realtime_update:
                update_text = 'TenClassesPyIO.setup_training_0001() progress %s/%s'%(str(i+1),str(data_size))
                print('%-64s'%(update_text),end='\r')
            _, cimg, _, variables = self.uniform_random_draw()
            self.x.append(cimg.transpose((2,0,1)))
            self.y.append(variables['y0'])
        print('%-64s'%('  data prepared.'))

    def setup_xai_evaluation_0001(self, general_meta_setting=None, explanation_setting=None, data_size=12, 
        realtime_update=False):
        self.x, self.y, self.h, self.v = [], [], [], []
        self.data_size = data_size

        self.setup0001(general_meta_setting, explanation_setting, s=None)
        for i in range(data_size):
            if realtime_update:
                update_text = 'TenClassesPyIO.setup_xai_evaluation_0001() progress %s/%s'%(str(i+1),str(data_size))
                print('%-64s'%(update_text),end='\r')
            _, cimg, heatmap, variables = self.uniform_random_draw()
            self.x.append(cimg)
            self.y.append(variables['y0'])
            self.h.append(heatmap)
            self.v.append(variables)
        print('%-64s'%('  data prepared.'))


class TenClassesPyIOwithHeatmap(TenClassesPyIO):
    def __init__(self, x,y,h, label_mapping={0:0., 1:0.4, 2:0.9}):
        """
        label_mapping is used to convert the values in heatmaps to labels
        Our standard values are 0,0.4,0.9 and we will label them as 0,1,2 respectively
        """
        super(TenClassesPyIOwithHeatmap, self).__init__()
        self.x = x
        self.y = y
        self.h = h
        self.data_size = len(self.x)
        self.label_mapping = label_mapping

    def __getitem__(self, index):   
        hx = np.array(self.h[index])
        h0 = (hx==self.label_mapping[0])*0 + (hx==self.label_mapping[1])*1 + (hx==self.label_mapping[2])*2

        return np.array(self.x[index]), np.array(self.y[index]), h0  