from comet_ml import Optimizer
import sys
import train
import os

params_file = "./parameters.json"

opt = Optimizer(sys.argv[1],
                trials=1,
                api_key="Bq3mQixNCv2jVzq2YBhLdxq9A")

for experiment in opt.get_experiments(project_name="fastdepth"):
    params = train.get_params(params_file)

#    try:
 #       pid =  os.environ["COMET_OPTIMIZER_PROCESS_ID"]
  #      params["device"] = pid
   # except KeyError:
    #    pass

    params["batch_size"] = experiment.get_parameter("batch_size")
    params["optimizer"]["lr"] = experiment.get_parameter("learning_rate")
    params["optimizer"]["momentum"] = experiment.get_parameter("momentum")
    params["optimizer"]["weight_decay"] = experiment.get_parameter("weight_decay")

    print("Batch Size: ", params["batch_size"])
    print("Learning Rate: " , params["optimizer"]["lr"])
    print("Momentum: ", params["optimizer"]["momentum"])
    print("Weight Decay: ", params["optimizer"]["weight_decay"])
    print("Device: ", params["device"])

    params, \
        train_loader, \
        val_loader, \
        test_loader, \
        model, \
        criterion, \
        optimizer = train.set_up_experiment(params, experiment)

    train.train(params, train_loader, val_loader,
          model, criterion, optimizer, experiment)

    train.evaluate(params, test_loader, model, criterion, experiment)

    experiment.end()
