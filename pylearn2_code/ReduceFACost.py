import theano.tensor as T
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin

class ReduceFACost(DefaultDataSpecsMixin, Cost):
    supervised = True

    def expr(self, model, data, **kwargs):
        space, source = self.get_data_specs(model)
        space.validate(data)

        inputs, targets = data
        outputs = model.mlp(inputs)

        # Need to modify
        loss = (targets * T.log(outputs)).sum(axis=1)
        return loss.mean()
