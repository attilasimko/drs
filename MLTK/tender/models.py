
class Tender(object): 
    def __init__(self, model, img): 
        import pygame
        import cv2
        from tensorflow.keras.models import Model, Sequential

        pygame.init()
        self.img = img
        self.model = model

        screen, px = Tender.setup(img)
        left, upper, right, lower = Tender.mainLoop(screen, px)

        # ensure output rect always has positive width, height
        if right < left:
            left, right = right, left
        if lower < upper:
            lower, upper = upper, lower
        pygame.display.quit()
        self.left, self.right, self.lower, self.upper = int(left/4), int(right/4), int(upper/4), int(lower/4)

    def pop_model(self, alpha):
        model = self.model
        left, right, lower, upper = self.left, self.right, self.lower, self.upper
        mask = np.ones_like(self.img[:, :, 0]) * alpha
        mask[left:right, lower:upper] = 1 

        if isinstance(model, Sequential):
            input_tensor = tensorflow.keras.Input(shape=(128, 128, 3))
            x = input_tensor
            for layer in model.layers:
                x = layer(x)
            model = Model(input_tensor, x)

        # The mask will be added to the following layers:
        # Conv2D

        input_tensor = tensorflow.keras.Input(shape=(128, 128, 3))
        x = input_tensor
        for layer in model.layers:
            x = layer(x)
            if Tender.layer_type(layer) == 'Conv2D':
                mask = cv2.resize(mask, 
                                  dsize=(x.shape[1], x.shape[2]),
                                  interpolation=cv2.INTER_NEAREST)
                mask = scipy.ndimage.gaussian_filter(mask, sigma=layer.kernel_size[0]/4)
                mask_for_layer = np.repeat(np.expand_dims(mask, axis=2), 
                                           x.shape[3], 
                                           axis=2)
                mask_for_layer = tensorflow.convert_to_tensor(mask_for_layer, dtype=tensorflow.float32)
                mask_for_layer = K.reshape(mask_for_layer, [-1, x.shape[1], x.shape[2], x.shape[3]])
                x = tensorflow.keras.layers.multiply([x, mask_for_layer])
                
            if Tender.layer_type(layer) == 'MaxPooling2D':
                mask = cv2.resize(mask, 
                                  dsize=(x.shape[1], x.shape[2]),
                                  interpolation=cv2.INTER_NEAREST)
                mask = scipy.ndimage.gaussian_filter(mask, sigma=layer.pool_size[0]/4)
                mask_for_layer = np.repeat(np.expand_dims(mask, axis=2), 
                                           x.shape[3], 
                                           axis=2)
                mask_for_layer = tensorflow.convert_to_tensor(mask_for_layer, dtype=tensorflow.float32)
                mask_for_layer = K.reshape(mask_for_layer, [-1, x.shape[1], x.shape[2], x.shape[3]])
                x = tensorflow.keras.layers.multiply([x, mask_for_layer])

        filename = "mask.png"
        plt.imshow(mask.T)
        plt.colorbar()
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close('all')
        return Model(input_tensor, x)

    def layer_type(layer):
        return str(layer)[10:].split(" ")[0].split(".")[-1]

    def plot_image(i, predictions_array, img):
        predictions_array, img = predictions_array[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        
        plt.imshow(np.array(img), cmap=plt.cm.binary)
        
        predicted_label = np.argmax(predictions_array)
        
        plt.xlabel("{} {:2.0f}%".format(class_names[predicted_label],
                                        100*np.max(predictions_array)))


    def plot_value_array(i, predictions_array):
        predictions_array = predictions_array[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        thisplot = plt.bar(range(2), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)


    def displayImage(screen, px, topleft, prior):
        # ensure that the rect always has positive width, height
        x, y = topleft
        width =  pygame.mouse.get_pos()[0] - topleft[0]
        height = pygame.mouse.get_pos()[1] - topleft[1]
        if width < 0:
            x += width
            width = abs(width)
        if height < 0:
            y += height
            height = abs(height)

        # eliminate redundant drawing cycles (when mouse isn't moving)
        current = x, y, width, height
        if not (width and height):
            return current
        if current == prior:
            return current

        # draw transparent box and blit it onto canvas
        screen.blit(px, px.get_rect())
        im = pygame.Surface((width, height))
        im.fill((128, 128, 128))
        pygame.draw.rect(im, (32, 32, 32), im.get_rect(), 1)
        im.set_alpha(128)
        screen.blit(im, (x, y))
        pygame.display.flip()

        # return current box extents
        return (x, y, width, height)

    def setup(cells):
        from PIL import Image
        cells = Image.fromarray(np.uint8(cells*255), 'RGB')
        cells = cells.resize((512, 512), Image.ANTIALIAS)
        cells = 255 * (np.array(cells)/ np.max(cells))
        # color dictionary, represents white, red and blue

        #create a surface with the size as the array
        surf = pygame.Surface((cells.shape[0], cells.shape[1]))
        # draw the array onto the surface
        pygame.surfarray.blit_array(surf, cells)
        # transform the surface to screen size
        screen = pygame.display.set_mode( surf.get_rect()[2:] )
        screen.blit(surf, surf.get_rect())
        pygame.display.flip()

        return screen, surf

    def mainLoop(screen, px):
        topleft = bottomright = prior = None
        n=0
        while n!=1:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONUP:
                    if not topleft:
                        topleft = event.pos
                    else:
                        bottomright = event.pos
                        n=1
            if topleft:
                prior = Tender.displayImage(screen, px, topleft, prior)
        return ( topleft + bottomright )