from .detectorX_template import DetectorX_template
import os
import torch
class IASSD_X(DetectorX_template):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        print('building IA-SSD cross modal')

    def forward(self, batch_dict):
        # if self.train_main:
        #     for cur_module in self.module_list:
        #         batch_dict = cur_module(batch_dict)
        # else:
        #     for cur_module in self.module_list:
        #         batch_dict = cur_module(batch_dict)
        #     batch_dict = self.module_list[self.head_idx]
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_point, tb_dict = self.point_head.get_loss()
        
        loss = loss_point
        return loss, tb_dict, disp_dict


    def load_backbone_params(self, filename, logger, to_cpu=False, backbone_id=0):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])

        backbone_key = 'multibackbone.module_list.' + str(backbone_id)

        update_model_state = {}
        for key, val in model_state_disk.items():
            if 'multibackbone.module_list.0' in key:
                model_key = key.replace('multibackbone.module_list.0', backbone_key)
            else:
                model_key = key
            if model_key in self.state_dict() and self.state_dict()[model_key].shape == model_state_disk[key].shape:
                if backbone_key in model_key:
                    # only load parameters of the selected backbone
                    update_model_state[model_key] = val
                    logger.info('Update weight %s: %s' % (model_key, str(val.shape)))

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        # for key in state_dict:
        #     if key not in update_model_state:
        #         logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))
