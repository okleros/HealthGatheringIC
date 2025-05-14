# Curious Agent

## Stable Baselines 3

```bash
uv run sb3/main.py
```

## Sample Factory

```bash 
uv run sf/main.py --env=health_gathering_glaucoma
```

⚠️ **Warning:**  In order to run the *play* action you need to change the line 218 of the python file learner
inside the sample-factory lib as below.

```py
checkpoint_dict = torch.load(latest_checkpoint, map_location=device)
```

to

```py
checkpoint_dict = torch.load(latest_checkpoint, map_location=device, weights_only=False)
```
