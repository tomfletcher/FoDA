Use the following Python code to read in the MNIST data:

data = np.load("mnist.npz")
x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']


The x data are 28x28 images, stored as 3D arrays. The first index tells you which image. The second and third index are the pixel coordinate. So, you will want to flatten these into row vectors in a data matrix. Use the following Python code:

n = x_train.shape[0]
X = x_train.reshape((n, 28*28))


Finally, to display an image, use the "imshow()" function of matplotlib on a 2D array. Something like:

plt.imshow(x_train[1,:,:])
plt.show()
