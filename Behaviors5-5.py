import numpy as np
import matplotlib.pyplot as plt
import cv2

# Please do not changes this section, they are parameters for threshold

ignore_threshold = 0.2 * 255
proximity_threshold = 0.25 * 255
np.seterr(all='raise')

# YOU CAN CHANGE THE FILENAMES HERE, PLEASE MAKE SURE THEY MATCHES
map_gt0 = np.array(np.load('GT_Info_Full_3.npy') * 255, dtype='uint8')
map_low0 = np.array(np.load('Belief_Info_Low_3.npy') * 255, dtype='uint8')

map_full0 = cv2.resize(map_low0, (2000, 2000), cv2.INTER_AREA)

# decide the size of the region-block map
Box_Size = 10
map_low0 = cv2.resize(np.array(map_low0, dtype='uint8'),
                      (int(2000 / Box_Size), int(2000 / Box_Size)), cv2.INTER_AREA)

edge_len = len(map_full0)
low_len = len(map_low0)

# specifying the size of allowed time windows before forced movement layer is activated
fm0 = 60
tolerance = 15
force_threshold = 0.4 * 255
distance_penalty = 1.5
erase_factor = 0.7

dir_towards = np.array([[0, 1], [-1, 1], [-1, 0],
                        [-1, -1], [0, -1], [1, -1],
                        [1, 0], [1, 1]], dtype='int8')


# find with region block does the location belongs to
def which_region(var1):
    return [int(var1[0] / Box_Size), int(var1[1] / Box_Size)]


# gathering information (e.g. take photos) at a specific location
def extraction(loc, region, map_low, map_full, map_gt):
    info = map_full[loc[0], loc[1]]
    info_gt = map_gt[loc[0], loc[1]]
    map_low[region[0], region[1]] -= info / (Box_Size ** 2)
    map_full[loc[0], loc[1]] = 0
    map_gt[loc[0], loc[1]] = 0
    return info, info_gt, map_low, map_full, map_gt


# loc_dest and loc_cache are points that are claimed to be satisfactory
def FPRF(loc_dest, loc_cache, map_low, map_full, map_gt):
    if loc_cache is None:
        print('This should not happen but anyway...')
        # This line should not be reached
        loc_cache = loc_dest

    if map_gt[loc_dest[0], loc_dest[1]] < proximity_threshold:
        region_dest = which_region(loc_dest)
        region_cache = which_region(loc_cache)
        region_len_x = int(abs(region_cache[0] - region_dest[0]) * erase_factor)
        region_len_y = int(abs(region_cache[1] - region_dest[1]) * erase_factor)

        region_x_start = max(0, region_dest[0] - region_len_x)
        region_x_end = min(len(map_low), region_dest[0] + region_len_x + 1)
        region_y_start = max(0, region_dest[1] - region_len_y)
        region_y_end = min(len(map_low), region_dest[1] + region_len_y + 1)

        map_low[region_x_start:region_x_end, region_y_start:region_y_end] = 0

        x_start = region_x_start * Box_Size
        x_end = region_x_end * Box_Size
        y_start = region_y_start * Box_Size
        y_end = region_y_end * Box_Size

        map_full[x_start:x_end, y_start:y_end] = 0
    return map_low, map_full


# calculating the penalized region block sum associated with each movement direction
def region_cal(trigonometric, map_origin, loc, visualization=False):
    up = np.argwhere(np.sin(3 * np.pi / 8) - 0.001 < trigonometric)  # up
    down = np.argwhere(trigonometric < np.sin(13 * np.pi / 8) + 0.0001)  # down
    a = np.argwhere((np.sin(-np.pi / 8) - 0.0001 < trigonometric) & (trigonometric < np.sin(np.pi / 8) + 0.0001))
    b = np.argwhere((np.sin(np.pi / 8) - 0.0001 <= trigonometric) & (trigonometric <= np.sin(3 * np.pi / 8) + 0.0001))
    c = np.argwhere(
        (np.sin(13 * np.pi / 8) - 0.0001 <= trigonometric) & (trigonometric <= np.sin(15 * np.pi / 8) + 0.0001))
    right = a[a[:, 1] > loc[1]]
    left = a[a[:, 1] < loc[1]]
    up_left = b[b[:, 1] < loc[1]]
    up_right = b[b[:, 1] > loc[1]]
    down_left = c[c[:, 1] < loc[1]]
    down_right = c[c[:, 1] > loc[1]]
    if visualization:
        # plotting the regions
        plt.axes()
        plt.xlim(0, 200)
        plt.ylim((200, 0))
        draw_region(up, 'blue')
        draw_region(down, 'red')
        draw_region(left, 'orange')
        draw_region(right, 'green')
        draw_region(up_left, 'purple')
        draw_region(up_right, 'violet')
        draw_region(down_left, 'grey')
        draw_region(down_right, 'black')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.draw()
        plt.show()

    direction_sum = []

    for i in [right, up_right, up, up_left, left, down_left, down, down_right]:
        sum0 = np.sum(map_origin[i[:, 0], i[:, 1]])
        direction_sum.append(sum0)

    return np.array(direction_sum, dtype='float')


# For visualization purpose only, disabled during trajectory planning.
def draw_region(direction, col='black'):
    for i in direction:
        # print(i)
        rectangle = plt.Rectangle((i[1] - 0.5, i[0] + 0.5), 1, 1, fc=col, ec=None, alpha=0.2)
        plt.gca().add_patch(rectangle)


# Layer 2 for self-directed roaming
def info_pot(loc, map_low):
    loc = which_region(loc)
    angles = np.array([[0] * low_len] * low_len, dtype='float16')
    map_temp = map_low.copy()
    for y in range(low_len):
        for x in range(low_len):
            if y == loc[0] and x == loc[1]:
                angles[loc[0], loc[1]] = 0

            else:
                height = loc[0] - y
                string = np.sqrt((loc[1] - x) ** 2 + (y - loc[0]) ** 2)
                angles[y, x] = height / string

                # preparing distance penalty
                map_temp[y, x] /= string ** distance_penalty

    return region_cal(angles, map_temp, loc)
    # prepare index in each direction


# implement proximity layer
def proximity(loc_proximity, map_gt, two_step=True):
    to_return = np.ones(8, dtype='float16')

    # if we have 5 by 5 vision for the UAV
    if two_step:
        # dir =0
        possible_steps = np.zeros((5, 5))
        y_min = loc_proximity[0] - 2
        y_max = loc_proximity[0] + 3
        x_min = loc_proximity[1] - 2
        x_max = loc_proximity[1] + 3
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                if y_min >= 0 and x_min >= 0 and x < 2000 and y < 2000:
                    possible_steps[y - y_min, x - x_min] = map_gt[y, x]

        possible_steps = possible_steps.flatten()
        all_combinations = [[9, 13, 14, 19], [3, 4, 8, 9], [1, 2, 3, 7], [0, 1, 5, 6], [5, 10, 11, 15],
                            [15, 16, 20, 21], [16, 17, 18, 22], [18, 19, 23, 24]]
        # dir 0
        to_return = np.zeros(8)
        for i in range(8):
            to_return[i] = np.sum(possible_steps[all_combinations[i]])

        return to_return

    # if we have 3 by 3 vision for the UAV
    else:

        if loc0[0] == 0:
            to_return[[1, 2, 3]] = 0
        elif loc0[0] == 1999:
            to_return[[5, 6, 7]] = 0
        if loc0[1] == 0:
            to_return[[3, 4, 5]] = 0
        elif loc0[1] == 1999:
            to_return[[0, 1, 7]] = 0

        indexes = np.argwhere(to_return == 1)

        for i in indexes:
            to_return[i] = map_gt[loc_proximity[0] + dir_towards[i, 0], loc_proximity[1] + dir_towards[i, 1]]

        return to_return


# change information map, and coordinate
def visit(directions0, loc, map_low, map_full, map_gt, chance, momentum0, fm):
    goto = np.random.choice(np.where(directions0 == directions0.max())[0])
    momentum0 = np.append(momentum0, [goto])
    fm -= 1

    # give some momentum to the UAV will make it less likely to move cyclically
    for i in momentum0:
        region = which_region(loc)
        loc = np.add(loc, dir_towards[i])
        loc = np.minimum([1999, 1999], loc)
        loc = np.maximum([0, 0], loc)
        info, info_gt, map_low, map_full, map_gt = extraction(loc, region, map_low, map_full, map_gt)

        # revert back to proximity layer if place with high information gain is reached
        if info_gt > proximity_threshold:
            fm = fm0
            chance = 8
            # print('loc= ', loc, ' direction= ', goto, ' info= ', info_gt)
            momentum0 = np.array([], dtype=int)
            break
        else:
            if info > proximity_threshold and abs(int(info_gt) - int(info)) > tolerance:
                chance = 0
                map_low[region[0], region[1]] = 0
                map_full[region[0] * Box_Size:region[0] * Box_Size + Box_Size,
                region[1] * Box_Size:region[1] * Box_Size + Box_Size] = 0

            else:
                chance -= 1

            if len(momentum0) >= 15 or chance > 0:
                momentum0 = np.delete(momentum0, 0)

    return loc, map_low, map_full, map_gt, chance, momentum0, fm


# marching towards the destination in a straight line
def force_visit(dir_prime, dir_sec, multiplicity, loc, map_low, map_full, map_gt):
    loc += np.array(dir_prime, dtype='int16')
    region = which_region(loc)
    myList = [np.array(loc, dtype='int16')]

    info, info_gt, map_low, map_full, map_gt = extraction(loc, region, map_low, map_full, map_gt)

    if info_gt < proximity_threshold:

        for i in range(multiplicity):

            loc += np.array(dir_sec, dtype='int16')
            myList.append(np.array(loc, dtype='int16'))
            region = which_region(loc)
            info, info_gt, map_low, map_full, map_gt = extraction(loc, region, map_low, map_full, map_gt)

            if info_gt >= proximity_threshold:
                print("breaking loc", loc)
                break
    return info, info_gt, myList, map_low, map_full, map_gt


def forced_move(loc, destination, map_low, map_full, map_gt):
    print('Start forced move:')
    print('current location', loc, 'destination', destination)
    delta_x = destination[0] - loc[0]
    delta_y = destination[1] - loc[1]
    info_gt = 0
    loc_record = []
    first_cache = None

    # go towared the pre-devided place unless somehwere satisfactory is found half-way to the destination coordinate
    while info_gt < proximity_threshold:
        direction_prime = [int(np.sign(delta_x)), int(np.sign(delta_y))]
        diff = np.abs(delta_x) - np.abs(delta_y)

        if delta_y == 0 or delta_x == 0:
            direction_secondary = direction_prime
            inter = np.abs(delta_x) - 1 if delta_x != 0 else np.abs(delta_y) - 1
        else:
            inter = int(np.floor(diff / np.abs(delta_y))) if diff > 0 else int(np.floor(diff / np.abs(delta_x)))
            direction_secondary = [int(np.sign(delta_x)), 0] if diff > 0 else [0, int(np.sign(delta_y))]

        info, info_gt, loc_temp, map_low, map_full, map_gt = \
            force_visit(direction_prime, direction_secondary, inter, loc, map_low, map_full, map_gt)

        loc = loc_temp[-1]
        print('current location:', loc, 'info=', info, 'real_info=', info_gt)
        if first_cache is None and info >= force_threshold * 0.64:
            first_cache = loc

        loc_record.extend(loc_temp)

        # deciding which would be the moving direction
        delta_x = destination[0] - loc[0]
        delta_y = destination[1] - loc[1]

        if delta_x == 0 and delta_y == 0:
            print('cached_loc=', first_cache)
            map_low, map_full = FPRF(destination, first_cache, map_low, map_full, map_gt)
            break

    return loc_record, map_low, map_full, map_gt


# deciding which would be the next policy to use and performe extraction.
def navigation(loc, map_low, map_full, map_gt, chance, momentum0, force_move0):
    if force_move0 < 0:
        # a relatively high force_threshold is set to make sure a new region are we heading to
        print('forced move situation encountered')
        region_unvisited = np.argwhere(map_low > force_threshold)

        # if no more information is believed to be in the region, terminate
        if len(region_unvisited) < 2:
            print('Stage clear. End condition met')
            np.save('trail.npy', path)
            exit()
        region_current = which_region(loc)
        dist = (region_unvisited[:, 0] - region_current[0]) ** 2 + (region_unvisited[:, 1] - region_current[1]) ** 2

        goto = region_unvisited[np.argmin(dist)] * Box_Size + np.array(Box_Size / 2, dtype='int16')
        loc, map_low, map_full, map_gt = forced_move(loc, goto, map_low, map_full, map_gt)
        force_move0 = fm0

    else:
        # If not forced Movement layer, decided if proximity or information potentioal
        direction = proximity(loc, map_gt) if chance >= 0 else info_pot(loc, map_low)

        loc, map_low, map_full, map_gt, chance, momentum0, force_move0 = \
            visit(direction, loc, map_low, map_full, map_gt, chance, momentum0, force_move0)
        loc = [np.array(loc, dtype='int16')]

    return loc, map_low, map_full, map_gt, chance, momentum0, force_move0


if __name__ == '__main__':

    momentum = np.array([], dtype='uint8')
    loc0 = np.array(np.random.rand(2) * 2000, dtype='uint32')
    steps, limit = 0, 800000
    force_move = fm0

    map_c = np.array(np.where(map_low0 > ignore_threshold, map_low0, 0), dtype='float16')
    map_f = np.array(np.where(map_full0 > ignore_threshold, map_full0, 0), dtype='uint8')
    map_g = map_gt0
    chance0 = 10
    path = []

    while steps <= limit:
        loc0, map_c, map_f, map_g, chance0, momentum, force_move = \
            navigation(loc0, map_c, map_f, map_g, chance0, momentum, force_move)

        path.extend(loc0)
        loc0 = loc0[-1]

        if steps % 500 == 0:
            print(steps, loc0)
        steps += 1

    np.save('trail.npy', path)
