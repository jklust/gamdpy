import numpy as np
import ipywidgets as widgets

class k3d_Visualization():
    def __init__(self, simulation):
        import k3d
        from k3d.colormaps import matplotlib_color_maps
        self.simulation = simulation
        self.conf = simulation.configuration
        # index 0 corresponds to 'r'
        self.plt_points = k3d.points(positions=self.conf['r'],
                            point_sizes=np.ones((self.conf.N),dtype=np.float32),
                            shader='mesh',
                            color_map=matplotlib_color_maps.Jet,
                            attribute=self.conf.scalars[:,0],
                            color_range=[-6, 0],
                            name='Atoms'
                           )
        Lx, Ly, Lz = self.conf.simbox.get_lengths()
   
        self.plt_box = k3d.lines(vertices=[ [-Lx/2, -Ly/2, -Lz/2], [-Lx/2, -Ly/2, +Lz/2], 
                                            [-Lx/2, +Ly/2, -Lz/2], [+Lx/2, -Ly/2, -Lz/2], 
                                            [-Lx/2, +Ly/2, +Lz/2], [+Lx/2, -Ly/2, +Lz/2], 
                                            [+Lx/2, +Ly/2, -Lz/2], [+Lx/2, +Ly/2, +Lz/2]], 
                                indices=[ [0,1], [0,2], [0,3], 
                                        [1,4], [1,5], [2,4], [2,6], [3,5], [3,6],
                                        [7,4], [7,5], [7,6]], 
                                indices_type='segment',
                                shader='mesh', width=float(min((Lx, Ly, Lz))/100), 
                                name='Simulation Box'
                                )
        self.plt_time_text = k3d.text2d('Time: ', position=[0.01, 0.15], is_html=True)
        self.plt_temp_text = k3d.text2d('Temperature: ', position=[0.01, 0.25], is_html=True,)
        self.plt_fn_text = k3d.text2d('Potential:', position=[0.01, 0.42], is_html=True)

        self.plot = k3d.plot(camera_mode='orbit', camera_fov=3.0,)
        self.plot += self.plt_points
        self.plot += self.plt_box
        self.plot += self.plt_fn_text + self.plt_time_text + self.plt_temp_text
        
        self.play = widgets.Play(
            value=self.simulation.num_blocks-1,
            min=0,
            max=self.simulation.num_blocks-1,
            step=1,
            interval=0,
            description="Press play",
            disabled=False
        )

        self.attribute_dropdown = widgets.Dropdown(
            options=[('Potential energy', 0), ('Virial', 1), ('Laplace U', 2), ('m', 3), ('Kinetic energy', 4), ('F^2', 5), ('Type', 6)],
            value=0,
            description='Color:',
            disabled=False,
        )

        self.slider = widgets.IntSlider(description='Frame:', max=self.play.max)
        widgets.jslink((self.play, 'value'), (self.slider, 'value'))

        self.w0 = widgets.interactive(self.update, block=self.play, choice=self.attribute_dropdown)
        self.w1 = widgets.interactive(self.set_color_range, choice=self.attribute_dropdown)


    def display(self):
        self.plot.display()

    def update(self, block, choice):
        restart_num = f"restart{block:04d}"
        # index 0 corresponds to 'r'
        self.plt_points.positions = self.simulation.output[f'/restarts/{restart_num}/vectors'][0]
        dt=self.simulation.output['/'].attrs['dt']
        model_time = (block+1)*dt*self.simulation.steps_per_block
        self.plt_time_text.text = f'Time: {model_time:.2f}'
        self.plt_temp_text.text = f'Temp: {self.simulation.integrator.temperature(model_time):.3f}'
        if choice==6:
            #self.plt_points.attribute = np.float32(self.conf.ptype)
            self.plt_points.attribute = np.float32(self.simulation.output[f'initial_configuration/ptype'])
        else:
            #self.plt_points.attribute = self.simulation.output[f'/restarts/{restart_num}/scalars'][:,choice]
            self.plt_points.attribute = self.simulation.output[f'/restarts/{restart_num}/scalars'][:,choice]
        Lx, Ly, Lz = self.simulation.output[f'/restarts/{restart_num}'].attrs['simbox_data']
        self.plt_box.vertices=[[-Lx/2, -Ly/2, -Lz/2], [-Lx/2, -Ly/2, +Lz/2], 
                      [-Lx/2, +Ly/2, -Lz/2], [+Lx/2, -Ly/2, -Lz/2], 
                      [-Lx/2, +Ly/2, +Lz/2], [+Lx/2, -Ly/2, +Lz/2], 
                      [+Lx/2, +Ly/2, -Lz/2], [+Lx/2, +Ly/2, +Lz/2]],

    def set_color_range(self, choice):
        labels = {0:'Potential:', 1:'Virial:', 2:'Laplace U:', 3:'m:', 4:'Kinetic nrg:', 5:'F^2', 6:'Type'}
        self.plt_fn_text.text = labels[choice]
        if choice == 6:
            minval, maxval = np.min(self.conf.ptype), np.max(self.conf.ptype)
        else:
            minval, maxval = np.percentile(self.simulation.scalars_list[:,:,choice], (1, 99))
        if minval==maxval:
            minval -= abs(minval*.1)
            maxval += abs(maxval*.1)
        self.plt_points.color_range = (minval, maxval)
    
    def display_player(self):
        display(widgets.HBox([self.play, self.slider, self.attribute_dropdown]))
