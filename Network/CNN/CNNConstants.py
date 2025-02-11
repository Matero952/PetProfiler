import torch
from torchvision.transforms import v2
from Network.CNN.model import CNN
import time

#Model

#Paths
MODEL_SAVE_PATH = f"../../results/parameters_{time.strftime('%Y%m%d')}"
TRAIN_JSON = "../../dataset/Train_Annotation/_annotations.coco.json"
VALID_JSON = "../../dataset/Valid_Annotation/_annotations.coco.json"
TEST_JSON = "../../dataset/Test_Annotation/_annotations.coco.json"
TRAIN_PLT_SAVE_PATH = f"../..results/train_graph{time.strftime('%Y%m%d')}"
VALID_PLT_SAVE_PATH = f"../..results/valid_graph{time.strftime('%Y%m%d')}"

LEARNING_RATE = 0.0000001
BATCH_SIZE = 4
EPOCHS = 1

#Safety features
EPOCH_PATIENCE = 5
WORST_TRAIN_LOSS = 0
WORST_VALID_LOSS = 0
PATIENCE_COUNTER = 0
BEST_TRAIN_CORRECT = 0
BEST_VALID_CORRECT = 0
ACCURACY_CUTOFF = 0.

#Device and specifics
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRANSFORMATION = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

#Sums and other variables
CORRECT_SUM = 0
INCORRECT_SUM = 0

#Matplot
TRAIN_CORRECT_LIST = []
TRAIN_INCORRECT_LIST = []
VALID_CORRECT_LIST = []
VALID_INCORRECT_LIST = []

TEST_CORRECT_LIST = []
TEST_INCORRECT_LIST = []

CNN_OUTPUT_SIZE = [2, 60, 319, 319]

#TODO Recalculate this using function we made in RPNUtility.



#K means scatter which will need to be redone in future when we incorporate unrecognized animal.
K_X = [361, 266.5, 547, 321, 533.5, 223, 470.5, 164, 482, 354, 275, 135.5, 610, 156.5, 373, 353.5, 123.5, 125.5, 113, 501.5, 373.5, 254, 460.5, 207.5, 600, 504.5, 171.5, 428.5, 235.5, 199.5, 320.5, 311.147, 276, 321.727, 521, 187.321, 255.945, 609.556, 264.238, 249.5, 574.916, 439.139, 396, 380.284, 200, 207.581, 384, 376.327, 226.5, 437, 367.337, 210.444, 483.263, 286.339, 228.327, 135.5, 169, 400.423, 179, 64, 246.5, 206.292, 228.5, 153.5, 187, 246.6, 186.134, 203, 221, 88.5, 144, 153, 185.5, 228.5, 206.804, 273.599, 155.072, 198.38, 71.812, 457.495, 107.668, 177.328, 364.171, 204, 198, 351.5, 245.5, 311.595, 247.23, 255, 308.718, 151.862, 124.455, 166.524, 340.5, 138.081, 367, 328.027, 331.377, 276, 383.474, 245, 329.913, 198.554, 327, 280, 139.397, 168, 273, 123.912, 357.289, 140.5, 373.5, 257, 151.303, 312, 514.469, 154.5, 148, 260.177, 159.73, 176.75, 308.469, 154.286, 292.82, 226.224, 332.362, 322.186, 229.758, 154.332, 128.5, 196.816, 171.5, 205.234, 142.63, 289.441, 249.5, 448.339, 158.5, 253, 195.175, 294.775, 369.274, 360.032, 273.813, 388.292, 213.43, 296.52, 346.472, 351.623, 306.19, 297.927, 355.5, 205.921, 397.534, 429.968, 209.158, 99.908, 349.707, 258.085, 80.503, 282.372, 343, 86.578, 101.5, 353.949, 315.5, 279.5, 426.543, 352.279, 229.5, 171.5, 273, 272, 372.775, 259.116, 322.069, 263.098, 192.124, 295.067, 362.578, 419.815, 349.713, 428.315, 203.5, 289.5, 105, 195.85, 155, 329.535, 189.657, 491, 391.5, 165, 124.5, 108, 198.326, 273.571, 227.364, 311.381, 145.4, 118.746, 247.483, 147.616, 137, 397.783, 252.853, 223.997, 286.52, 389.775, 254.448, 215.067, 83, 140.5, 433.329, 179.146, 475.854, 290, 297.395, 102, 81, 223.5, 130, 240, 175.909, 251.316, 325, 254.749, 370, 330.416, 425.261, 387.8, 350.5, 125.714, 203.489, 198.93, 275.5, 206, 369.449, 128, 515.419, 263.983, 137.5, 235, 251.786, 281.425, 419.276, 119.5, 288, 423, 359.681, 346.688, 306.823, 180.404, 197.496, 129.288, 25.184, 269, 295.835, 241.5, 411.114, 345.795, 198.758, 240.538, 323.543, 410.006, 349, 238, 437.261, 266.606, 339.975, 156.789, 198.46, 288.768, 362, 343.613, 424.886, 306, 400.98, 312.5, 208.5, 160.153, 254.125, 423.5, 279.77, 451, 49.5, 116.365, 71.829, 280.287, 336.97, 252, 205.5, 511, 211.82, 157.803, 394.461, 274.591, 283.109, 372.746, 195.175, 289.979, 337.34, 409.835, 250.239, 325.109, 517.995, 422.485, 254.376, 230.992, 371.185, 257.35, 315.688, 304.776, 155.214, 137, 213, 154, 147.578, 183.5, 167.425, 362.202, 311, 369.43, 277.224, 299.257, 251.897, 215.687, 226.911, 161.305, 201.591, 150.793, 177.261, 534.883, 376.351, 253.466, 230.5, 387.924, 256.197, 309.5, 247.124, 295.1, 207, 334.957, 144, 118.5, 277.938, 308.63, 122, 266.728, 200.5, 125.49, 338, 236, 210.5, 175.5, 399, 107, 285.5, 357, 475, 343, 212.5, 159, 134.5, 85.5, 333, 431.5, 529, 151.5, 219, 535.5, 141, 371, 299.5, 465.5, 397, 484.5, 257, 365, 494]
K_Y = [362, 333, 596, 270, 570, 333, 571, 243, 554, 269, 269.5, 261.5, 556, 213, 290.5, 152.5, 102.5, 121, 300, 510.5, 322, 408.5, 475, 155.5, 508, 524, 215.5, 538.5, 221, 213, 256.5, 515.917, 264.5, 302.468, 169.5, 292.076, 318.7, 557.088, 287.717, 120.5, 543.227, 289.265, 191.5, 283.06, 230, 312.018, 188.5, 353.142, 301.5, 205.5, 304.309, 141.21, 272.447, 210.917, 121.927, 206, 259.806, 253.701, 271.5, 123.5, 242, 222.742, 385, 73, 234, 421.697, 260.173, 286, 231.5, 301, 156.5, 251, 254.5, 186.5, 325.534, 100.045, 258.322, 301.286, 135.755, 291.554, 166, 246, 256.814, 406.5, 236.5, 170, 95, 241.354, 119.769, 337, 325.796, 252.071, 130.012, 231.677, 346, 200.55, 373.234, 173.838, 341.607, 212, 375.104, 258, 437.449, 299.617, 135, 183.5, 162.419, 322.5, 142.5, 132.6, 314.772, 212.5, 244.5, 189, 231.861, 296, 335.238, 201, 153.5, 398.42, 67.913, 244.082, 314.613, 229.911, 241.712, 303.735, 277.627, 259.481, 239.641, 240.594, 216, 306.282, 188, 286.439, 230.889, 353, 510, 330.006, 283, 118, 326.425, 204.057, 421.978, 248.448, 289.323, 420.992, 181.964, 288.8, 262.865, 134.602, 207.936, 240.408, 140.5, 226.706, 205.503, 369.52, 331.613, 148.283, 224.635, 275.992, 259.327, 495.764, 221.5, 240.431, 131, 331.027, 107, 239, 223.314, 158.943, 264.5, 303.5, 182, 112.5, 197.641, 113.835, 162.172, 167.06, 241.15, 220.003, 525.082, 292.038, 323.768, 229.727, 244, 168, 186.5, 168.036, 269.5, 283.591, 295.591, 624.334, 177.5, 193.5, 222, 109.5, 309.15, 104.231, 252.35, 196.589, 240.154, 124.756, 487.39, 245.65, 253, 400.292, 169.062, 242.842, 164.887, 375.259, 322.874, 237.023, 153, 225.5, 217.776, 249.216, 356.133, 113.5, 357.494, 115.5, 83, 143.5, 275.5, 319, 255.034, 206.245, 176, 237.964, 209.5, 389.066, 286.073, 197.999, 251, 169.955, 254.194, 330.012, 234.5, 292.5, 272.583, 206.5, 571.045, 338.808, 220.5, 200, 307, 245, 261.461, 213.5, 288.5, 278, 344.343, 172.028, 234.88, 346.048, 209.018, 136.431, 90.314, 326.748, 268.041, 137.5, 317.86, 340.248, 315.397, 255.575, 217.904, 335.755, 166, 68.5, 271.507, 374.312, 274.347, 289.381, 260.175, 150.456, 316, 324.611, 338.899, 299, 342.308, 155, 238, 242.755, 221.591, 189.5, 130.287, 536, 234.5, 130.216, 80.441, 379, 218, 163.5, 119, 511, 248.054, 190.483, 270.563, 264.571, 351.612, 392.279, 325.354, 250.54, 330.544, 282.273, 375.367, 200.844, 242.772, 286.28, 189.454, 244.245, 192.762, 110.102, 357.584, 309.489, 226.618, 447, 258.921, 202.173, 212.547, 158, 254.668, 327.089, 285, 298.019, 178.918, 367.716, 242.195, 285.222, 407.084, 68.775, 249.369, 314.739, 175.045, 303.476, 218.083, 101.622, 199.5, 241.074, 117.677, 234.5, 162.734, 194.132, 340, 170, 294.5, 210.5, 344.619, 261.698, 260, 304.711, 236.5, 214.716, 327.5, 122, 189, 246, 307, 95, 220.5, 193, 224.5, 151.5, 397, 319, 167.5, 97, 231, 461.5, 460, 204, 276, 529, 205, 265, 166, 229.5, 201.5, 452.5, 198.5, 265, 178]

K_MEANS_PLT_SAVE_PATH = "/home/dante/Doggy_Door/Plots/K_Means_Plot.png"

IDEAL_ANCH_WIDTH = 207.10
IDEAL_ANCH_HEIGHT = 212.61

ASPECT_RATIOS = [[1, 3], [2, 3], [3, 4], [9,6]]

stride = 11
#IMPORTANT!!!! WHEN I WAS REFACTORING THIS FILE AND CHANGED THIS TO STRIDE, IT FUCKED THE WHOLE ANCHORS
#FILE UP. I THINK ITS BECAUSE STRIDE IS BUILT IN AS AN ATTRIBUTE FOR PYTORCH. KEEP IT ALL LOWERCASE PLEAS FUTURE MATEO.
print(type(stride))


