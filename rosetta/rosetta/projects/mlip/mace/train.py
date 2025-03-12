import time
from typing import Callable

import cuequivariance as cue
import cuequivariance_jax as cuex
import flax
import flax.linen
import jax
import jax.numpy as jnp
import numpy as np
import optax
from cuequivariance.experimental.mace import symmetric_contraction
from cuequivariance_jax.experimental.utils import MultiLayerPerceptron, gather


class MACELayer(flax.linen.Module):
    first: bool
    last: bool
    num_species: int
    num_features: int  # typically 128
    interaction_irreps: cue.Irreps  # typically 0e+1o+2e+3o
    hidden_irreps: cue.Irreps  # typically 0e+1o
    activation: Callable  # typically silu
    epsilon: float  # typically 1/avg_num_neighbors
    max_ell: int  # typically 3
    correlation: int  # typically 3
    output_irreps: cue.Irreps  # typically 1x0e
    readout_mlp_irreps: cue.Irreps  # typically 16x0e
    replicate_original_mace_sc: bool = True
    skip_connection_first_layer: bool = False

    @flax.linen.compact
    def __call__(
        self,
        vectors: cuex.IrrepsArray,  # [num_edges, 3]
        node_feats: cuex.IrrepsArray,  # [num_nodes, irreps]
        node_species: jax.Array,  # [num_nodes] int between 0 and num_species-1
        radial_embeddings: jax.Array,  # [num_edges, radial_embedding_dim]
        senders: jax.Array,  # [num_edges]
        receivers: jax.Array,  # [num_edges]
    ):
        dtype = node_feats.dtype

        def lin(irreps: cue.Irreps, input: cuex.IrrepsArray, name: str):
            e = cue.descriptors.linear(input.irreps(), irreps)
            w = self.param(name, jax.random.normal, (e.inputs[0].irreps.dim,), dtype)
            return cuex.equivariant_tensor_product(e, w, input, precision="HIGH")

        def linZ(irreps: cue.Irreps, input: cuex.IrrepsArray, name: str):
            e = cue.descriptors.linear(input.irreps(), irreps)
            w = self.param(
                name,
                jax.random.normal,
                (self.num_species, e.inputs[0].irreps.dim),
                dtype,
            )
            # Dividing by num_species for consistency with the 1-hot implementation
            return cuex.equivariant_tensor_product(
                e, w[node_species], input, precision="HIGH"
            ) / jnp.sqrt(self.num_species)

        if True:
            i = jnp.argsort(receivers)
            senders = senders[i]
            receivers = receivers[i]
            vectors = vectors[i]
            radial_embeddings = radial_embeddings[i]
            indices_are_sorted = True
            del i
        else:
            indices_are_sorted = False

        if self.last:
            hidden_out = self.hidden_irreps.filter(keep=self.output_irreps)
        else:
            hidden_out = self.hidden_irreps

        self_connection = None
        if not self.first or self.skip_connection_first_layer:
            self_connection = linZ(
                self.num_features * hidden_out, node_feats, "linZ_skip_tp"
            )

        node_feats = lin(node_feats.irreps(), node_feats, "linear_up")

        messages = node_feats[senders]
        sph = cuex.spherical_harmonics(range(self.max_ell + 1), vectors)
        e = cue.descriptors.channelwise_tensor_product(
            messages.irreps(), sph.irreps(), self.interaction_irreps
        )
        e = e.squeeze_modes().flatten_coefficient_modes()

        mix = MultiLayerPerceptron(
            [64, 64, 64, e.inputs[0].irreps.dim],
            self.activation,
            output_activation=False,
            with_bias=False,
        )(radial_embeddings)

        messages = cuex.equivariant_tensor_product(
            e,
            mix,
            messages,
            sph,
            algorithm="compact_stacked" if e.all_same_segment_shape() else "sliced",
        )

        node_feats = gather(
            receivers,
            messages,
            node_feats.shape[0],
            indices_are_sorted=indices_are_sorted,
        )
        node_feats *= self.epsilon

        node_feats = lin(
            self.num_features * self.interaction_irreps, node_feats, "linear_down"
        )

        # This is only used in the first layer if it has no skip connection
        if self.first and not self.skip_connection_first_layer:
            # Selector TensorProduct
            node_feats = linZ(
                self.num_features * self.interaction_irreps,
                node_feats,
                "linZ_skip_tp_first",
            )

        e, projection = symmetric_contraction(
            node_feats.irreps(),
            self.num_features * hidden_out,
            range(1, self.correlation + 1),
        )
        n = projection.shape[0 if self.replicate_original_mace_sc else 1]
        w = self.param(
            "symmetric_contraction",
            jax.random.normal,
            (self.num_species, n, self.num_features),
            dtype,
        )
        if self.replicate_original_mace_sc:
            w = jnp.einsum("zau,ab->zbu", w, projection)
        w = jnp.reshape(w, (self.num_species, -1))

        node_feats = cuex.equivariant_tensor_product(
            e,
            w[node_species],
            node_feats,
            algorithm="compact_stacked" if e.all_same_segment_shape() else "sliced",
        )

        node_feats = lin(self.num_features * hidden_out, node_feats, "linear_post_sc")

        if self_connection is not None:
            node_feats = node_feats + self_connection

        node_outputs = node_feats
        if self.last:  # Non linear readout for last layer
            assert self.readout_mlp_irreps.is_scalar()
            assert self.output_irreps.is_scalar()
            node_outputs = cuex.scalar_activation(
                lin(self.readout_mlp_irreps, node_outputs, "linear_mlp_readout"),
                self.activation,
            )
        node_outputs = lin(self.output_irreps, node_outputs, "linear_readout")

        return node_outputs, node_feats


class radial_basis(flax.linen.Module):
    r_max: float
    num_radial_basis: int
    num_polynomial_cutoff: int = 5

    def envelope(self, x: jax.Array) -> jax.Array:
        p = float(self.num_polynomial_cutoff)
        xs = x / self.r_max
        xp = jnp.power(xs, self.num_polynomial_cutoff)
        return (
            1.0
            - 0.5 * (p + 1.0) * (p + 2.0) * xp
            + p * (p + 2.0) * xp * xs
            - 0.5 * p * (p + 1.0) * xp * xs * xs
        )

    def bessel(self, x: jax.Array) -> jax.Array:
        bessel_weights = (
            jnp.pi / self.r_max * jnp.arange(1, self.num_radial_basis + 0.5)
        )
        prefactor = jnp.sqrt(2.0 / self.r_max)
        numerator = jnp.sin(bessel_weights * x)
        return prefactor * (numerator / x)

    @flax.linen.compact
    def __call__(self, edge: jax.Array) -> jax.Array:
        assert edge.ndim == 0
        cutoff = jnp.where(edge < self.r_max, self.envelope(edge), 0.0)
        radial = self.bessel(edge)
        return radial * cutoff


class MACEModel(flax.linen.Module):
    num_layers: int
    num_features: int
    num_species: int
    max_ell: int
    correlation: int
    num_radial_basis: int
    interaction_irreps: cue.Irreps
    hidden_irreps: cue.Irreps
    offsets: np.ndarray
    cutoff: float
    epsilon: float

    @flax.linen.compact
    def __call__(
        self,
        vecs: jax.Array,  # [num_edges, 3]
        species: jax.Array,  # [num_nodes] int between 0 and num_species-1
        senders: jax.Array,  # [num_edges]
        receivers: jax.Array,  # [num_edges]
        graph_index: jax.Array,  # [num_nodes]
        num_graphs: int,
    ) -> tuple[jax.Array, jax.Array]:
        def model(vecs):
            with cue.assume(cue.O3, cue.ir_mul):
                w = self.param(
                    "linear_embedding",
                    jax.random.normal,
                    (self.num_species, self.num_features),
                    vecs.dtype,
                )
                node_feats = cuex.as_irreps_array(
                    w[species] / jnp.sqrt(self.num_species)
                )

                radial_embeddings = jax.vmap(
                    radial_basis(self.cutoff, self.num_radial_basis)
                )(jnp.linalg.norm(vecs, axis=1))
                vecs = cuex.IrrepsArray("1o", vecs)

                Es = 0
                for i in range(self.num_layers):
                    first = i == 0
                    last = i == self.num_layers - 1
                    output, node_feats = MACELayer(
                        first=first,
                        last=last,
                        num_species=self.num_species,
                        num_features=self.num_features,
                        interaction_irreps=self.interaction_irreps,
                        hidden_irreps=self.hidden_irreps,
                        activation=jax.nn.silu,
                        epsilon=self.epsilon,
                        max_ell=self.max_ell,
                        correlation=self.correlation,
                        output_irreps=cue.Irreps(cue.O3, "1x0e"),
                        readout_mlp_irreps=cue.Irreps(cue.O3, "16x0e"),
                    )(vecs, node_feats, species, radial_embeddings, senders, receivers)
                    Es += jnp.squeeze(output.array, 1)
                return jnp.sum(Es), Es

        Fterms, Ei = jax.grad(model, has_aux=True)(vecs)
        offsets = jnp.asarray(self.offsets, dtype=Ei.dtype)
        Ei = Ei + offsets[species]

        E = jnp.zeros((num_graphs,), Ei.dtype).at[graph_index].add(Ei)

        nats = jnp.shape(species)[0]
        F = (
            jnp.zeros((nats, 3), Ei.dtype)
            .at[senders]
            .add(Fterms)
            .at[receivers]
            .add(-Fterms)
        )

        return E, F


def main():
    # Dataset specifications
    num_species = 50
    num_graphs = 100
    num_atoms = 4_000
    num_edges = 70_000
    avg_num_neighbors = 20

    # Model specifications
    model = MACEModel(
        num_layers=2,
        num_features=128,
        num_species=num_species,
        max_ell=3,
        correlation=3,
        num_radial_basis=8,
        interaction_irreps=cue.Irreps(cue.O3, "0e+1o+2e+3o"),
        hidden_irreps=cue.Irreps(cue.O3, "0e+1o"),
        offsets=np.zeros(num_species),
        cutoff=5.0,
        epsilon=1 / avg_num_neighbors,
    )

    # Dummy data
    vecs = jax.random.normal(jax.random.key(0), (num_edges, 3))
    species = jax.random.randint(jax.random.key(0), (num_atoms,), 0, num_species)
    senders, receivers = jax.random.randint(
        jax.random.key(0), (2, num_edges), 0, num_atoms
    )
    graph_index = jax.random.randint(jax.random.key(0), (num_atoms,), 0, num_graphs)
    graph_index = jnp.sort(graph_index)
    target_E = jax.random.normal(jax.random.key(0), (num_graphs,))
    target_F = jax.random.normal(jax.random.key(0), (num_atoms, 3))

    # Initialization
    w = jax.jit(model.init, static_argnums=(6,))(
        jax.random.key(0), vecs, species, senders, receivers, graph_index, num_graphs
    )
    opt = optax.adam(1e-2)
    opt_state = opt.init(w)

    # Training
    @jax.jit
    def step(
        w,
        opt_state,
        vecs: jax.Array,
        species: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        graph_index: jax.Array,
        target_E: jax.Array,
        target_F: jax.Array,
    ):
        def loss_fn(w):
            E, F = model.apply(
                w, vecs, species, senders, receivers, graph_index, target_E.shape[0]
            )
            return jnp.mean((E - target_E) ** 2) + jnp.mean((F - target_F) ** 2)

        grad = jax.grad(loss_fn)(w)
        updates, opt_state = opt.update(grad, opt_state)
        w = optax.apply_updates(w, updates)
        return w, opt_state

    for i in range(100):
        t0 = time.perf_counter()
        w, opt_state = step(
            w,
            opt_state,
            vecs,
            species,
            senders,
            receivers,
            graph_index,
            target_E,
            target_F,
        )

        jax.block_until_ready(w)
        t1 = time.perf_counter()
        print(f"{i:04}: {t1-t0:.3}s")

        break


if __name__ == "__main__":
    main()