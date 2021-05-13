import numpy as np
import matplotlib.pyplot as plt
import cv2


def run():
    first_time = True

    print('idx multiplicity %map_visited ROI_coverage FN FP eff')

    # YOU CAN SPECIFY THE INDEX OF THE IMAGES YOU WANT TO PLOT HERE
    for idx in [3]:  # range(20):

        background_raw = cv2.imread('GT_Info_Full_' + str(idx) + '.png', 0)

        for multiplicity in [0]:
            # This line may need to be changed for other dataset
            path = np.load('trail.npy')

            # This line may need to be changed for other dataset
            belief_map = cv2.imread('Belief_Info_Low_' + str(idx) + '.png', 0)

            y = path[:, 0]
            x = path[:, 1]

            exploration = np.zeros((2000, 2000), dtype='uint8')
            belief_full = cv2.resize(belief_map, (2000, 2000))
            background = background_raw

            for shift1 in range(-2, 3):
                for shift2 in range(-2, 3):
                    temp_y = np.where(y >= 0, y + shift2, 0)
                    temp_y = np.where(temp_y < 2000, temp_y, 1999)

                    temp_x = np.where(x >= 0, x + shift1, 0)
                    temp_x = np.where(temp_x < 2000, temp_x, 1999)
                    exploration[temp_y, temp_x] = background[temp_y, temp_x]
            if first_time:
                plt.figure(figsize=(7, 7), dpi=200)

                plt.subplot(2, 2, 1)
                plt.gca().set_aspect('equal', adjustable='box')

                # Use background if working on real img
                plt.imshow(background_raw, cmap='gray')
                plt.plot(x, y, ls='', marker='.', ms=0.03)
                plt.ylim(2000, 0)  # decreasing time
                plt.xlim(0, 2000)
                plt.title('trajectory path')

                # need to de-noising
                plt.subplot(2, 2, 2)
                # Use background  if working on real img
                plt.imshow(background_raw, cmap='gray')
                plt.title('ground truth')

                plt.subplot(2, 2, 3)
                plt.imshow(belief_full, cmap='gray')
                plt.title('belief map')

                plt.subplot(2, 2, 4)
                plt.imshow(exploration, cmap='gray')
                plt.title('explored region')
                plt.gca().set_aspect('equal', adjustable='box')

                plt.savefig('visualization.png')
                plt.show()

            edge_len = 50

            GT_temp = np.array(cv2.resize(background, (edge_len, edge_len)), dtype='float')

            belief_map = np.array(belief_map, dtype='float')

            info_gathered = exploration.sum() / background.sum()

            relative_efficiency = exploration.sum() / len(path) / 255

            FN = np.where((GT_temp - belief_map) > 0.2 * 255, 1, 0).sum() / len(np.argwhere(GT_temp != 0))
            FP = np.where((belief_map - GT_temp) > 0.2 * 255, 1, 0).sum() / len(np.argwhere(belief_map > 25))

            print(idx, multiplicity, len(path) / 2000 ** 2, info_gathered, FN, FP, relative_efficiency)


if __name__ == "__main__":
    run()
