from condensate import Data  
import matplotlib.pyplot as plt

spinupdata = Data.import(loc='/data_1')

print(spinupdata.atom)
print(spinupdata.sequence)

spinupdata.images=[img1, img2, etc]

spinupdata.times=[0, 1e-3, 2e-3, ...]

testimage = spinupdata.images[0]

f, axarr = plt.subplots(ncols=2)
axarr[0].imshow(testimage.density)
axarr[0].set_title('density t={:.2f}'.format(testimage.time))
axarr[1].imshow(testimage.phase)
axarr[1].set_title('phase')
plt.show()

for i in len(spinupdata.images):
    image = spinupdata.images[i]
    np.sum(image.density, axis=1)
    # fit widths, append to dataframe?

plt.plot(results)
    

