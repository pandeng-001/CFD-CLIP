# Plot loss curves

import re
import matplotlib.pyplot as plt

# load loss.txt
with open('./log/(CLIP)resnet-101/txt/8-(CLIP)resnet101-classifier__trian_losses.txt', 'r') as f:
    loss_data = f.read().strip().split(', ')

# load contra.txt
with open('./log/(CLIP)resnet-101/txt/8-(CLIP)resnet101-classifier__trian_contra_losses.txt', 'r') as f:
    contra_content  = f.read().strip()

# Use regex to match all items (including full tensor structures)
contra_items = re.findall(r"tensor\([^)]+\)|[0-9\.]+", contra_content)


# Trim leading/trailing whitespace from each item
contra_data = [item.strip() for item in contra_items]

# Extract numerical values from contra_values
contra_values = []
for item in contra_data:
    # Extract values from tensors
    tensor_match = re.search(r'tensor\(([\d.]+)', item)
    if tensor_match:
        contra_values.append(float(tensor_match.group(1)))
    else:
        # Convert regular numerical values directly
        contra_values.append(float(item))

# Ensure loss and contra have consistent lengths
min_length = min(len(loss_data), len(contra_values))
loss_data = [float(x) for x in loss_data[:min_length]]
contra_values = contra_values[:min_length]

# Calculate loss1 and loss2
loss1 = [float(loss_data[i]) - 8*contra_values[i] for i in range(len(loss_data))]
loss2 = [8*contra_values[i] for i in range(len(contra_values)) if contra_values[i] !=0 ]
print(loss1[:60])
print(loss2[:60])

# plot 
plt.figure(figsize=(10, 5))

plt.plot(loss1, label='CEloss')
plt.plot(loss2, label=' Distillation Loss')

plt.xlabel('Epochs')
plt.ylabel('Training loss')
plt.title('Training loss')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig("./plot_img/img/resnet18_img_loss.png")
