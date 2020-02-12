from abc import abstractmethod
import configparser
import numpy as np
from argparse import ArgumentParser
import keras
import tensorflow as tf
from util import create_instance, load_class, ParserAble
import itertools


class Prediction(ParserAble):

    @abstractmethod
    def __call__(self, y_pred):
        pass


class MaxPrediction(Prediction):
    def __call__(self, y_pred):
        return np.argmax(y_pred, axis=1)


class IndividualPrediction(MaxPrediction):

    def get_parser(self) -> ArgumentParser:
        parser = super().get_parser()
        parser.add_argument('--nb_classes', type=int, required=True)
        parser.add_argument('--k', type=int, required=True)
        parser.add_argument('--return_prob', action='store_true')
        return parser

    def __call__(self, y_pred):
        FLAGS = self.params
        nb_classes = FLAGS.nb_classes
        k = FLAGS.k
        n_outs = np.prod(y_pred.shape[1:])
        label = np.ones((y_pred.shape[0], (n_outs // nb_classes) * k), dtype=np.int) * -1
        probs = np.ones((y_pred.shape[0], (n_outs // nb_classes) * k), dtype=np.float32) * -1
        for j in range(y_pred.shape[0]):
            for i in range(n_outs // nb_classes):
                if i == 0:
                    idxs = np.argsort(y_pred[j, :nb_classes])[-k:]
                    label[j, i * k:(i + 1) * k] = idxs
                    probs[j, i * k:(i + 1) * k] = y_pred[j, idxs + i * nb_classes]
                else:
                    idxs = np.argsort(y_pred[j, i * nb_classes:(i + 1) * nb_classes])[-k:]
                    label[j, i * k:(i + 1) * k] = idxs
                    probs[j, i * k:(i + 1) * k] = y_pred[j, idxs + i * nb_classes]
        if FLAGS.return_prob:
            return np.concatenate((label, probs), axis=1)
        return label


class AggregatedPrediction(MaxPrediction):

    def get_parser(self) -> ArgumentParser:
        parser = super().get_parser()
        parser.add_argument('--nb_classes', type=int, required=True)
        return parser

    def __call__(self, y_pred):
        FLAGS = self.params
        n_outs = np.prod(y_pred.shape[1:])
        label = None
        nb_classes = FLAGS.nb_classes
        for i in range(n_outs // nb_classes):
            if label is None:
                label = super().__call__(y_pred[:, :nb_classes])
            else:
                l_tmp = super().__call__(y_pred[:, i * nb_classes:(i + 1) * nb_classes])
                # -2 means incosistent output across the models
                label[label != l_tmp] = -2
        return label


class BaseAttack(ParserAble):
    def to_dict(self, vals, keys: list):
        rows = []
        n = vals.shape[0]
        m = vals.shape[1]
        for i in range(n):
            row = {}
            for j in range(m):
                row[keys[j]] = vals[i, j]
            rows.append(row)
        return rows

    @abstractmethod
    def generate(self, x, **kwargs):
        pass


class ModelLoader(ParserAble):
    def get_softmax_layer(self, model: keras.models.Model):
        n_layers = len(model.layers)
        layers = model.layers
        for i in range(n_layers):
            l = layers[n_layers - 1 - i]
            if isinstance(l, keras.layers.Activation) or isinstance(l, keras.layers.Dense):
                return l
        model.summary()
        raise Exception('Softmax layer not found')

    def get_logits(self, model: keras.models.Model = None) -> keras.models.Model:
        if model is None:
            model = self.get_model()
        last = self.get_softmax_layer(model)
        last.activation = keras.activations.linear
        model_copy = keras.models.clone_model(model)
        model_copy.set_weights(model.get_weights())
        return model_copy

    @abstractmethod
    def get_model(self) -> keras.models.Model:
        pass

    def get_info_dict(self) -> dict:
        return vars(self.params)


class PretrainedModelSetLoader(ModelLoader):

    def get_parser(self) -> ArgumentParser:
        parser = super().get_parser()
        parser.add_argument('--model_id', type=lambda x: x.split(';'), required=True)
        parser.add_argument('--input_shape', type=lambda x: tuple(map(int, x.split(";"))), required=True)
        return parser

    def get_pp_fn_name(self, model_id):
        return model_id.rsplit(".", 1)[0] + ".preprocess_input"

    def get_model(self) -> keras.models.Model:
        FLAGS = self.params
        pp_fn_names = list(map(self.get_pp_fn_name, FLAGS.model_id))
        inp = keras.layers.Input(shape=FLAGS.input_shape)
        outputs = []
        for m_id, pp_fn_name in zip(FLAGS.model_id, pp_fn_names):
            model_fn = load_class(m_id)
            pre_process_fn = load_class(pp_fn_name)
            model = model_fn(weights='imagenet')
            pp_inp = keras.layers.Lambda(lambda x: pre_process_fn(x))(inp)
            out = model(pp_inp)
            outputs.append(out)
        concated = keras.layers.concatenate(outputs, axis=-1)
        final_model = keras.models.Model(inputs=inp, outputs=concated)
        return final_model

    def get_logits(self) -> keras.models.Model:
        FLAGS = self.params
        pp_fn_names = list(map(self.get_pp_fn_name, FLAGS.model_id))
        inp = keras.layers.Input(shape=FLAGS.input_shape)
        outs = []
        for m_id, pp_fn_name in zip(FLAGS.model_id, pp_fn_names):
            model_fn = load_class(m_id)
            pre_process_fn = load_class(pp_fn_name)
            model = model_fn(weights='imagenet')
            pp_inp = keras.layers.Lambda(lambda x: pre_process_fn(x))(inp)
            model_logits = super().get_logits(model)
            out = model_logits(pp_inp)
            outs.append(out)
        concated = keras.layers.concatenate(outs, axis=-1)
        final_model = keras.models.Model(inputs=inp, outputs=concated)
        return final_model


class StepFunction(ParserAble):

    @abstractmethod
    def get_direction(self, sample, a_grad, t_grad, a_pred_vals, t_pred_vals, consider, clip_min, clip_max):
        pass


class MaxStep(StepFunction):

    def get_direction(self, sample, a_grad, t_grad, a_pred_vals, t_pred_vals, consider, clip_min, clip_max):
        w_norm, w = -1, None
        for i in range(consider.shape[0]):
            if not consider[i]:
                continue
            w_i = t_grad[i] - a_grad[i]
            norm_i = np.linalg.norm(w_i.flatten(), ord=2)
            if norm_i < 1e-4:
                raise Exception('Vanishing gradient')
            p_i = abs(t_pred_vals[i] - a_pred_vals[i]) / norm_i
            if p_i > w_norm:
                w_norm = p_i + 1e-4
                w = w_i / norm_i
        return w_norm * w


class TargetFunction(ParserAble):
    @abstractmethod
    def __init__(self, config: configparser.ConfigParser, args, src_sec_name=None, static_flag=False,
                 stats_flag=False) -> None:
        super().__init__(config, args, src_sec_name)
        self.static_flag = static_flag
        self.stats_flag = stats_flag

    def is_static(self):
        return self.static_flag

    def has_stats(self):
        return self.stats_flag

    def get_stat_names(self, n_models):
        raise NotImplementedError()

    def get_stats(self):
        raise NotImplementedError()

    def get_product_ranking(self, pred_row, n_models):
        n_classes = pred_row.shape[0] // n_models
        model_outs = np.zeros((n_models, n_classes))
        for i in range(n_models):
            model_outs[i] = pred_row[i * n_classes:(i + 1) * n_classes]
        idxs = np.argsort(-model_outs, axis=1)
        ranking = np.zeros((n_models, n_classes))
        ranks = np.arange(n_classes)
        for i in range(n_models):
            ranking[i] = ranks[np.argsort(idxs[i])]
        aggregated_ranking = np.ones_like(ranks)
        for i in range(n_models):
            aggregated_ranking = aggregated_ranking * ranking[i]
        aggregated_idxs = np.argsort(aggregated_ranking)
        return aggregated_idxs

    @abstractmethod
    def gen_target_states(self, original, current, prediction_vals, grads_t, grad_idx_ph, x, x_adv, sess: tf.Session,
                          nb_classes):
        pass


class ConsistentTarget(TargetFunction):

    def __init__(self, config: configparser.ConfigParser, args, src_sec_name=None) -> None:
        super().__init__(config, args, src_sec_name, False)
        self.static_flag = self.params.static

    def is_static(self):
        return self.static_flag

    def get_parser(self) -> ArgumentParser:
        parser = super().get_parser()
        parser.add_argument('--nb_candidates', type=int, required=True)
        parser.add_argument('--static', action='store_true')
        parser.add_argument('--reverse', action='store_true')
        return parser

    def gen_target_states(self, original, current, prediction_vals, grads_t, grad_idx_ph, x, x_adv, sess: tf.Session,
                          nb_classes):
        n_samples = original.shape[0]
        n_models = nb_classes.shape[0]
        nb_candidates = self.params.nb_candidates
        is_reverse = self.params.reverse
        target_states = np.zeros((n_samples, nb_candidates, n_models), dtype=np.int)

        for i in range(n_samples):
            pred_row = prediction_vals[i]
            ranking = self.get_product_ranking(pred_row, n_models)
            ranking = ranking[ranking != original[i]]
            for j in range(nb_candidates):
                if is_reverse:
                    target_states[i, j, :] = ranking[-(j + 1)]
                else:
                    target_states[i, j, :] = ranking[j]

        return target_states


class DiverseTarget(TargetFunction):

    def __init__(self, config: configparser.ConfigParser, args, src_sec_name=None) -> None:
        super().__init__(config, args, src_sec_name, static_flag=True, stats_flag=True)
        self.stats = None

    def get_stat_names(self, n_models):
        field_names = []
        for i in range(n_models - 1):
            for j in range(i + 1, n_models):
                field_names.append("cossim_" + str(i) + "vs" + str(j))
        return field_names

    def get_stats(self):
        return self.stats

    def get_parser(self) -> ArgumentParser:
        parser = super().get_parser()
        parser.add_argument('--nb_candidates', type=int, required=True)
        return parser

    def get_diversity(self, vectors):
        n = vectors.shape[0]
        similiraties = np.ones((n - 1) * n // 2) * -1000
        k = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                # these vectors must have unit length
                cossim = np.sum(vectors[i] * vectors[j])
                similiraties[k] = cossim
                k = k + 1
        return similiraties

    def gen_target_states(self, original, current, prediction_vals, grads_t, grad_idx_ph, x, x_adv, sess: tf.Session,
                          nb_classes):
        n_samples = original.shape[0]
        n_models = nb_classes.shape[0]
        self.stats = np.zeros((n_samples, (n_models - 1) * n_models // 2))
        nb_candidates = self.params.nb_candidates
        target_states = np.zeros((n_samples, nb_candidates, n_models), dtype=np.int)
        idx_holder = np.zeros((n_models * n_samples, 2), dtype=np.int)
        current_idx_shift = np.zeros_like(nb_classes, dtype=np.int)
        current_idx_shift[1:] = np.cumsum(nb_classes[:-1])

        for i in range(n_samples):
            # set actual idxs
            idx_holder[i * n_models:(i + 1) * n_models, 0] = i
            idx_holder[i * n_models:(i + 1) * n_models, 1] = original[i] + current_idx_shift

            # calculate rankings and store last nb_candidate classidxs
            pred_row = prediction_vals[i]
            ranking = self.get_product_ranking(pred_row, n_models)
            ranking = ranking[ranking != original[i]]
            for j in range(nb_candidates):
                target_states[i, j, :] = ranking[-(j + 1)]
        grad_a = sess.run(grads_t, feed_dict={grad_idx_ph: idx_holder, x: x_adv})
        grad_ts = np.zeros((nb_candidates, n_models, n_samples) + x_adv.shape[1:])
        for i in range(nb_candidates):
            target_idxs = target_states[:, i] + current_idx_shift
            idx_holder[:, 1] = target_idxs.flatten()
            grad_ts[i] = sess.run(grads_t, feed_dict={grad_idx_ph: idx_holder, x: x_adv})
            grad_ts[i] = grad_ts[i] - grad_a
            for j in range(n_models):
                for k in range(n_samples):
                    grad_ts[i, j, k] = grad_ts[i, j, k] / np.linalg.norm(grad_ts[i, j, k].flatten(), ord=2)
        final_states = np.zeros((n_samples, 1, n_models), dtype=np.int)
        directions = np.zeros((n_models,) + x_adv.shape[1:], dtype=np.float32)
        possible_labels = [list(range(nb_candidates))] * n_models
        for i in range(n_samples):
            min_v = 1
            for selected_targets in itertools.product(*possible_labels):
                for k in range(n_models):
                    directions[k] = grad_ts[selected_targets[k], k, i]
                similarities = self.get_diversity(directions)
                a_v = np.mean(similarities)
                if a_v < min_v:
                    self.stats[i] = similarities
                    min_v = a_v
                    for k in range(n_models):
                        final_states[i, 0, k] = target_states[i, selected_targets[k], k]

        return final_states


class RandomTarget(TargetFunction):

    def __init__(self, config: configparser.ConfigParser, args, src_sec_name=None) -> None:
        super().__init__(config, args, src_sec_name=src_sec_name, static_flag=True)
        FLAGS = self.params
        self.rand_gen = np.random.RandomState(FLAGS.seed)

    def get_parser(self) -> ArgumentParser:
        parser = super().get_parser()
        parser.add_argument('--seed', type=int, default=9)
        return parser

    def gen_target_states(self, original, current, prediction_vals, grads_t, grad_idx_ph, x, x_adv, sess: tf.Session,
                          nb_classes):
        n_samples = original.shape[0]
        n_models = nb_classes.shape[0]
        target_states = np.zeros((n_samples, 1, n_models), dtype=np.int)
        for i in range(n_samples):
            for j in range(n_models):
                rint = self.rand_gen.random_integers(0, nb_classes[j] - 2, size=1)
                if rint == current[i, j]:
                    rint = (rint + 1) % nb_classes[j]
                target_states[i, 0, j] = rint
        return target_states


class CustomTarget(TargetFunction):

    def __init__(self, config: configparser.ConfigParser, args, src_sec_name=None) -> None:
        super().__init__(config, args, src_sec_name=src_sec_name, static_flag=True)

    def get_parser(self) -> ArgumentParser:
        parser = super().get_parser()
        parser.add_argument('--pattern', type=lambda x: list(map(int, x.split(";"))), required=True)
        return parser

    def gen_target_states(self, original, current, prediction_vals, grads_t, grad_idx_ph, x, x_adv, sess: tf.Session,
                          nb_classes):
        n_samples = original.shape[0]
        n_models = nb_classes.shape[0]
        pattern = self.params.pattern
        target_states = np.zeros((n_samples, 1, n_models), dtype=np.int)
        for i in range(n_samples):
            for j in range(n_models):
                target_states[i, 0, j] = pattern[j]
        return target_states


class ModelSetAttack(BaseAttack):

    def __init__(self, config: configparser.ConfigParser, args, src_sec_name=None) -> None:
        super().__init__(config, args, src_sec_name)
        FLAGS = self.params
        if not isinstance(FLAGS.target_func, TargetFunction):
            raise Exception("target_func must be a TargetFunction class")
        self.model_holder = create_instance(FLAGS.src_model, self.config, self.args)
        if not isinstance(self.model_holder, ModelLoader):
            raise Exception("src_model must be a Model Loader class")
        self.model_logits = self.model_holder.get_logits()
        self.sess = keras.backend.get_session()
        self.x_adv_tensor = None
        self.l_orig_tensor = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.field_names = []
        if FLAGS.target_func.has_stats():
            self.field_names += FLAGS.target_func.get_stat_names(FLAGS.n_models)
        for i in range(FLAGS.n_models):
            self.field_names.append('model_out_' + str(i))
        self.field_names.append('succeed')
        self.field_names.append('n_iter')

    def get_parser(self) -> ArgumentParser:
        parser = super().get_parser()
        parser.add_argument('--src_model', type=str, required=True)
        parser.add_argument('--pred_func', type=lambda x: create_instance(x, self.config, self.args),
                            required=True)
        parser.add_argument('--target_func', type=lambda x: create_instance(x, self.config, self.args),
                            required=True)
        parser.add_argument('--step_func', type=lambda x: create_instance(x, self.config, self.args),
                            required=True)
        parser.add_argument('--max_iter', type=int, default=100)
        parser.add_argument('--n_models', type=int, required=True)
        parser.add_argument('--nb_classes', type=lambda x: np.array(list(map(int, x.split(';')))), required=True)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--clip_max', type=float, default=1)
        parser.add_argument('--max_step', type=float, default=np.inf)
        parser.add_argument('--min_step', type=float, default=1e-4)
        parser.add_argument('--clip_min', type=float, default=0)
        parser.add_argument('--overshoot', type=float, default=0.02)

        return parser

    def get_labels(self, prediction_vals, nb_classes):
        n_samples = prediction_vals.shape[0]
        n_models = nb_classes.shape[0]
        out = np.ones((n_samples, n_models), dtype=np.int) * -1
        n_outs = np.cumsum(nb_classes, dtype=np.int)
        for i in range(n_samples):
            row = prediction_vals[i]
            for j in range(n_models):
                if j == 0:
                    out[i, j] = np.argmax(row[0:n_outs[j]])
                else:
                    out[i, j] = np.argmax(row[n_outs[j - 1]:n_outs[j]])
        return out

    def is_not_succeed(self, current, target_states):
        n_samples, nb_candidates = target_states.shape[0], target_states.shape[1]
        not_succeed = np.ones(n_samples, dtype=np.bool)
        for i in range(n_samples):
            row = current[i]
            target_row = target_states[i]
            for j in range(nb_candidates):
                if np.min(row == target_row[j]) == 1:
                    not_succeed[i] = 0
                    break
        return not_succeed

    def get_tensor(self, max_iter, clip_max, clip_min, overshoot, target_func: TargetFunction, nb_classes, max_step,
                   min_step, step_func: StepFunction):
        if self.x_adv_tensor is not None:
            return self.x_adv_tensor
        x = self.model_logits.input
        l_orig = self.l_orig_tensor
        logits = self.model_logits.output
        pred_vals = logits
        grad_idx = tf.placeholder(dtype=tf.int32, shape=(None, 2))
        n_models = len(nb_classes)
        grads = []
        selected_outputs = tf.gather_nd(pred_vals, grad_idx)
        for i in range(n_models):
            grad = tf.gradients(selected_outputs[i::n_models], x)
            grads.append(grad[0])
        grads = tf.stack(grads)

        # Define graph
        def deepfool_wrap(x_val, l_orig):
            x_adv = np.array(x_val, copy=True, dtype=np.float32)
            stats = np.zeros((x_val.shape[0], len(self.field_names)), dtype=np.float32)
            iteration = 0
            prediction_vals = self.sess.run(logits, feed_dict={x: x_val})
            current = self.get_labels(prediction_vals, nb_classes)
            current_idx_shift = np.zeros_like(nb_classes, dtype=np.int)
            current_idx_shift[1:] = np.cumsum(nb_classes[:-1])
            target_states = target_func.gen_target_states(l_orig, current, prediction_vals, grads, grad_idx, x, x_adv,
                                                          self.sess,
                                                          nb_classes)
            not_succeed = self.is_not_succeed(current, target_states)
            print('l_orig: ', l_orig)
            n_samples, n_target_states = x_val.shape[0], target_states.shape[1]

            idx_holder = np.zeros((n_models * n_samples, 2), dtype=np.int)
            for j in range(1, n_samples):
                idx_holder[j * n_models:(j + 1) * n_models, 0] = j
            selected_states = np.ones((n_samples, n_models)) * -1
            a_pert = np.zeros_like(x_val)
            a_norm = np.ones(n_samples) * -1

            while np.any(not_succeed) and iteration < max_iter:
                current_idxs = current + current_idx_shift
                idx_holder[:, 1] = current_idxs.flatten()
                a_grad = self.sess.run(grads, feed_dict={x: x_adv, grad_idx: idx_holder})

                for j in range(n_target_states):
                    target_idxs = target_states[:, j] + current_idx_shift
                    idx_holder[:, 1] = target_idxs.flatten()
                    t_grad = self.sess.run(grads, feed_dict={x: x_adv, grad_idx: idx_holder})
                    for i in range(n_samples):
                        if not not_succeed[i]:
                            continue
                        stats[i, -1] = iteration
                        prediction_vals_row = prediction_vals[i]
                        current_row_idxs = current_idxs[i]
                        target_row_idxs = target_idxs[i]
                        a_row_grad = a_grad[:, i]
                        t_row_grad = t_grad[:, i]
                        w_t = step_func.get_direction(x_adv[i], a_row_grad, t_row_grad,
                                                      prediction_vals_row[current_row_idxs],
                                                      prediction_vals_row[target_row_idxs],
                                                      current_row_idxs != target_row_idxs, clip_min=clip_min,
                                                      clip_max=clip_max)
                        norm_t = np.linalg.norm(w_t.flatten(), ord=2)
                        if norm_t < a_norm[i] or j == 0:
                            a_norm[i] = np.clip(norm_t, min_step, max_step)
                            a_pert[i] = a_norm[i] / norm_t * w_t
                            selected_states[i] = target_states[i, j]
                print('Pert:', np.mean(a_norm[not_succeed]), np.min(a_norm[not_succeed]), np.max(a_norm[not_succeed]))
                x_adv[not_succeed] = np.clip(x_adv[not_succeed] + a_pert[not_succeed], clip_min, clip_max)
                dist = np.sqrt(np.sum(np.square(x_adv - x_val), axis=tuple(range(1, len(x_val.shape)))))
                print('Dist:', np.mean(dist[not_succeed]), np.min(dist[not_succeed]), np.max(dist[not_succeed]))

                prediction_vals = self.sess.run(logits, feed_dict={x: x_adv})
                current = self.get_labels(prediction_vals, nb_classes)
                if not target_func.is_static():
                    target_states = target_func.gen_target_states(l_orig, current, prediction_vals, grads, grad_idx,
                                                                  x, x_adv,
                                                                  self.sess,
                                                                  nb_classes)
                not_succeed = self.is_not_succeed(current, target_states)
                print("Attack result at iteration {0} is {1} selected states {2}".format(iteration, current,
                                                                                         selected_states))
                print("{0} out of {1} become adversarial examples at iteration {2}".format(
                    np.sum(np.logical_not(not_succeed)),
                    x_adv.shape[0],
                    iteration))
                iteration += 1
            r_tot = x_adv - x_val
            n_attack_fields = n_models + 2
            if target_func.has_stats():
                stats[:, :-n_attack_fields] = target_func.get_stats()
            stats[:, -2] = np.logical_not(not_succeed) * 1
            stats[:, -n_attack_fields:-2] = current
            # need to clip this image into the given range
            x_adv = np.clip((1 + overshoot) * r_tot + x_val, clip_min, clip_max)
            return [x_adv, stats]

        wrap = tf.py_func(deepfool_wrap, [x, l_orig], (tf.float32, tf.float32))
        self.x_adv_tensor = wrap
        return self.x_adv_tensor

    def get_nb_classes(self):
        FLAGS = self.params
        if FLAGS.nb_classes.shape[0] != FLAGS.n_models:
            nb_classes = np.zeros(FLAGS.n_models)
            nb_classes[:] = FLAGS.nb_classes[0]
            FLAGS.nb_classes = nb_classes
        return FLAGS.nb_classes

    def generate(self, x, **kwargs):
        FLAGS = self.params
        p_label = FLAGS.pred_func(self.model_logits.predict(x, batch_size=FLAGS.batch_size))
        x_adv_tens = self.get_tensor(FLAGS.max_iter,
                                     FLAGS.clip_max,
                                     FLAGS.clip_min,
                                     FLAGS.overshoot,
                                     target_func=FLAGS.target_func,
                                     nb_classes=self.get_nb_classes(),
                                     max_step=FLAGS.max_step,
                                     min_step=FLAGS.min_step,
                                     step_func=FLAGS.step_func
                                     )
        res = self.sess.run(x_adv_tens, feed_dict={self.model_logits.input: x, self.l_orig_tensor: p_label})
        x_adv = res[0]

        adv_stats = res[1]

        return x_adv, self.to_dict(adv_stats, self.field_names)
