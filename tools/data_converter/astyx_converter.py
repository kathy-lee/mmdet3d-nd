import mmcv
import numpy as np
import json


from pathlib import Path
from mmdet3d.core.bbox import box_np_ops
from concurrent import futures as futures
from object3d_astyx import Object3dAstyx, inv_trans
from skimage import io


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def create_astyx_info_file(data_path,
                           pkl_prefix='astyx',
                           save_path=None,
                           relative_path=True,
                           pc_type='lidar'):
    """Create info file of Astyx dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        data_path (str): Path of the data root.
        pkl_prefix (str): Prefix of the info file to be generated.
        save_path (str): Path to save the info file.
        relative_path (bool): Whether to use relative path.
    """
    imageset_folder = Path(data_path) / 'ImageSets'
    train_img_ids = _read_imageset_file(str(imageset_folder / 'train.txt'))
    val_img_ids = _read_imageset_file(str(imageset_folder / 'val.txt'))
    test_img_ids = _read_imageset_file(str(imageset_folder / 'test.txt'))

    print('Generate info. this may take several minutes.')
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)
    astyx_infos_train = get_astyx_image_info(
        data_path,
        training=True,
        velodyne=True,
        calib=True,
        image_ids=train_img_ids,
        relative_path=relative_path,
        pc_type=pc_type)
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'Kitti info train file is saved to {filename}')
    mmcv.dump(astyx_infos_train, filename)
    kitti_infos_val = get_astyx_image_info(
        data_path,
        training=True,
        velodyne=True,
        calib=True,
        image_ids=val_img_ids,
        relative_path=relative_path,
        pc_type=pc_type)
    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'Astyx info val file is saved to {filename}')
    mmcv.dump(kitti_infos_val, filename)
    filename = save_path / f'{pkl_prefix}_infos_trainval.pkl'
    print(f'Astyx info trainval file is saved to {filename}')
    mmcv.dump(astyx_infos_train + kitti_infos_val, filename)

    astyx_infos_test = get_astyx_image_info(
        data_path,
        training=False,
        label_info=False,
        velodyne=True,
        calib=True,
        image_ids=test_img_ids,
        relative_path=relative_path,
        pc_type=pc_type)
    filename = save_path / f'{pkl_prefix}_infos_test.pkl'
    print(f'Astyx info test file is saved to {filename}')
    mmcv.dump(astyx_infos_test, filename)


def _create_reduced_point_cloud(data_path,
                                info_path,
                                save_path=None,
                                back=False,
                                num_features=4,
                                front_camera_id=2):
    """Create reduced point clouds for given info.

    Args:
        data_path (str): Path of original data.
        info_path (str): Path of data info.
        save_path (str | None): Path to save reduced point cloud data.
            Default: None.
        back (bool): Whether to flip the points to back.
        num_features (int): Number of point features. Default: 4.
        front_camera_id (int): The referenced/front camera ID. Default: 2.
    """
    astyx_infos = mmcv.load(info_path)

    for info in mmcv.track_iter_progress(astyx_infos):
        pc_info = info['point_cloud']
        image_info = info['image']
        calib = info['calib']

        v_path = pc_info['velodyne_path']
        v_path = Path(data_path) / v_path
        points_v = np.fromfile(
            str(v_path), dtype=np.float32,
            count=-1).reshape([-1, num_features])
        rect = calib['R0_rect']
        if front_camera_id == 2:
            P2 = calib['P2']
        else:
            P2 = calib[f'P{str(front_camera_id)}']
        Trv2c = calib['Tr_velo_to_cam']
        # first remove z < 0 points
        # keep = points_v[:, -1] > 0
        # points_v = points_v[keep]
        # then remove outside.
        if back:
            points_v[:, 0] = -points_v[:, 0]
        points_v = box_np_ops.remove_outside_points(points_v, rect, Trv2c, P2,
                                                    image_info['image_shape'])
        if save_path is None:
            save_dir = v_path.parent.parent / (v_path.parent.stem + '_reduced')
            if not save_dir.exists():
                save_dir.mkdir()
            save_filename = save_dir / v_path.name
            # save_filename = str(v_path) + '_reduced'
            if back:
                save_filename += '_back'
        else:
            save_filename = str(Path(save_path) / v_path.name)
            if back:
                save_filename += '_back'
        with open(save_filename, 'w') as f:
            points_v.tofile(f)


def create_reduced_point_cloud(data_path,
                               pkl_prefix,
                               train_info_path=None,
                               val_info_path=None,
                               test_info_path=None,
                               save_path=None,
                               with_back=False):
    """Create reduced point clouds for training/validation/testing.

    Args:
        data_path (str): Path of original data.
        pkl_prefix (str): Prefix of info files.
        train_info_path (str | None): Path of training set info.
            Default: None.
        val_info_path (str | None): Path of validation set info.
            Default: None.
        test_info_path (str | None): Path of test set info.
            Default: None.
        save_path (str | None): Path to save reduced point cloud data.
        with_back (bool): Whether to flip the points to back.
    """
    if train_info_path is None:
        train_info_path = Path(data_path) / f'{pkl_prefix}_infos_train.pkl'
    if val_info_path is None:
        val_info_path = Path(data_path) / f'{pkl_prefix}_infos_val.pkl'
    if test_info_path is None:
        test_info_path = Path(data_path) / f'{pkl_prefix}_infos_test.pkl'

    print('create reduced point cloud for training set')
    _create_reduced_point_cloud(data_path, train_info_path, save_path)
    print('create reduced point cloud for validation set')
    _create_reduced_point_cloud(data_path, val_info_path, save_path)
    print('create reduced point cloud for testing set')
    _create_reduced_point_cloud(data_path, test_info_path, save_path)
    if with_back:
        _create_reduced_point_cloud(
            data_path, train_info_path, save_path, back=True)
        _create_reduced_point_cloud(
            data_path, val_info_path, save_path, back=True)
        _create_reduced_point_cloud(
            data_path, test_info_path, save_path, back=True)


def get_astyx_image_info(path,
                         training=True,
                         label_info=True,
                         velodyne=False,
                         calib=False,
                         image_ids=0,
                         extend_matrix=True,
                         num_worker=8,
                         relative_path=True,
                         with_imageshape=True,
                         pc_type='lidar'):
    """
    Astyx annotation format:
    {
        point_cloud: {
            num_features: ...
            velodyne_path: ...
        }
        [optional]points: [N, 3+] point cloud
        [optional]image: {
            image_idx: ...
            image_path: ...
            image_shape: ...
        }
        [optional]calib: {
            R0_rect: ...
            Tr_velo_to_cam: ...
            P2: ...
        }
        [optional]annos: {
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_y: [num_gt] angle array
            name: [num_gt] ground truth name array
            [optional]difficulty: kitti difficulty
            [optional]group_ids: used for multi-part object
        }
    }
    """
    root_path = Path(path)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))

    def map_func(idx):
        info = {}

        pc_info = {'num_features': 4, 'pc_idx': idx}
        info['point_cloud'] = pc_info

        image_info = {'image_idx': idx, 'image_shape': get_image_shape(root_path, idx)}
        info['image'] = image_info

        calib = get_calib(root_path, idx)
        info['calib'] = calib

        if label_info:
            obj_list = get_label(root_path, idx)
            for obj in obj_list:
                obj.from_radar_to_camera(calib)
                obj.from_radar_to_image(calib)
                ###############################
                # print(f'\n%s:' % sample_idx)
                #
                # # print(obj.loc, obj.orient)
                # # obj.from_lidar_to_radar(calib)
                # # print(obj.loc, obj.orient)
                #
                # # print(obj.loc_camera, obj.rot_camera)
                # # obj.from_lidar_to_camera(calib)
                # # print(obj.loc_camera, obj.rot_camera)
                #
                # print(obj.box2d)
                # obj.from_lidar_to_image(calib)
                # print(obj.box2d)
                ###############################

            annotations = {'name': np.array([obj.cls_type for obj in obj_list]),
                           'occluded': np.array([obj.occlusion for obj in obj_list]),
                           'alpha': np.array([-np.arctan2(obj.loc[1], obj.loc[0])
                                              + obj.rot_camera for obj in obj_list]),
                           'bbox': np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0),
                           'dimensions': np.array([[obj.l, obj.h, obj.w] for obj in obj_list]),
                           'location': np.concatenate([obj.loc_camera.reshape(1, 3) for obj in obj_list], axis=0),
                           'rotation_y': np.array([obj.rot_camera for obj in obj_list]),
                           'score': np.array([obj.score for obj in obj_list]),
                           'difficulty': np.array([obj.level for obj in obj_list], np.int32),
                           'truncated': -np.ones(len(obj_list))}

            num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
            num_gt = len(annotations['name'])
            index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
            annotations['index'] = np.array(index, dtype=np.int32)
            if pc_type == 'radar':
                gt_boxes = np.array([[*obj.loc, obj.w, obj.l, obj.h, obj.rot] for obj in obj_list])
            else:
                for obj in obj_list:
                    obj.from_radar_to_lidar(calib)
                gt_boxes = np.array([[*obj.loc_lidar, obj.w, obj.l, obj.h, obj.rot_lidar] for obj in obj_list])
            annotations['gt_boxes'] = gt_boxes

            points = get_pointcloud(idx, pc_type)
            indices = box_np_ops.points_in_rbbox(points[:, :3], gt_boxes)
            num_points_in_gt = indices.sum(0)
            num_ignored = num_gt - num_objects
            num_points_in_gt = np.concatenate(
                [num_points_in_gt, -np.ones([num_ignored])])
            annotations['num_points_in_gt'] = num_points_in_gt

        if annotations is not None:
            info['annos'] = annotations
            # add_difficulty_to_annos(info)
        return info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, image_ids)

    return list(image_infos)


def get_image_shape(root_path, idx):
    img_file = root_path / 'camera_front' / ('%s.jpg' % idx)
    assert img_file.exists()
    return np.array(io.imread(img_file).shape[:2], dtype=np.int32)


def get_label(root_path, idx):
    label_file = root_path / 'groundtruth_obj3d' / ('%s.json' % idx)
    assert label_file.exists()
    with open(label_file, 'r') as f:
        data = json.load(f)
    objects = [Object3dAstyx.from_label(obj) for obj in data['objects']]
    return objects


def get_calib(root_path, idx):
    calib_file = root_path / 'calibration' / ('%s.json' % idx)
    assert calib_file.exists()
    with open(calib_file, 'r') as f:
        data = json.load(f)

    T_from_lidar_to_radar = np.array(data['sensors'][1]['calib_data']['T_to_ref_COS'])
    T_from_camera_to_radar = np.array(data['sensors'][2]['calib_data']['T_to_ref_COS'])
    K = np.array(data['sensors'][2]['calib_data']['K'])
    T_from_radar_to_lidar = inv_trans(T_from_lidar_to_radar)
    T_from_radar_to_camera = inv_trans(T_from_camera_to_radar)
    return {'T_from_radar_to_lidar': T_from_radar_to_lidar,
            'T_from_radar_to_camera': T_from_radar_to_camera,
            'T_from_lidar_to_radar': T_from_lidar_to_radar,
            'T_from_camera_to_radar': T_from_camera_to_radar,
            'K': K}


def get_pointcloud(root_path, idx, pc_type):
    if pc_type == 'lidar':
        lidar_file = root_path / 'lidar_vlp16' / ('%s.txt' % idx)
        assert lidar_file.exists()
        return np.loadtxt(str(lidar_file), dtype=np.float32, skiprows=1, usecols=(0, 1, 2, 3))
    elif pc_type == 'radar':
        radar_file = root_path / 'radar_6455' / ('%s.txt' % idx)
        assert radar_file.exists()
        #return np.loadtxt(str(radar_file), dtype=np.float32, skiprows=2, usecols=(0, 1, 2, 3, 4))
        pc = np.loadtxt(str(radar_file), dtype=np.float32, skiprows=2, usecols=(0, 1, 2, 3, 4))
        pc[:, -1] -= 45
        return pc
    else:
        # for point cloud fusion
        pass
