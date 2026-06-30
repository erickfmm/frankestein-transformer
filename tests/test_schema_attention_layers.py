import pathlib
import unittest

from src.utils.schema_loader import resolve_schema


class SchemaAttentionLayerTests(unittest.TestCase):
    def test_schema_includes_all_attention_layer_names(self):
        schema_path = pathlib.Path(__file__).parent.parent / "src" / "schema.yaml"
        schema = resolve_schema(schema_path)

        enum_values = (
            schema["properties"]["model"]["properties"]["layer_pattern"]["items"]["enum"]
        )

        expected = {
            "retnet",
            "retnet_attn",
            "mamba",
            "ode",
            "titan_attn",
            "standard_attn",
            "sigmoid_attn",
            "sparse_transformer_attn",
            "longformer_attn",
            "bigbird_attn",
            "sparsek_attn",
            "nsa_attn",
            "sparge_attn",
            "fasa_attn",
            "gla_attn",
            "deltanet_attn",
            "gated_deltanet_attn",
            "gated_deltanet2_attn",
            "hgrn2_attn",
            "fox_attn",
            "gated_softmax_attn",
            "engram_attn",
            "gqa_attn",
            "msa_attn",
            "sparda_attn",
            "kda_attn",
            "mla_attn",
            "gqla_attn",
            "mlra_attn",
            "tucker_attn",
            "iha_attn",
            "gta_attn",
            "mtla_attn",
        }

        self.assertTrue(expected.issubset(set(enum_values)))

    def test_schema_includes_mixture_of_depths_fields(self):
        schema_path = pathlib.Path(__file__).parent.parent / "src" / "schema.yaml"
        schema = resolve_schema(schema_path)

        model_properties = schema["properties"]["model"]["properties"]

        for field_name in [
            "use_mixture_of_depths",
            "mixture_of_depths_capacity_ratio",
            "mixture_of_depths_router_aux_loss_weight",
        ]:
            self.assertIn(field_name, model_properties)

    def test_schema_includes_new_latent_sparse_gated_fields(self):
        schema_path = pathlib.Path(__file__).parent.parent / "src" / "schema.yaml"
        schema = resolve_schema(schema_path)

        model_properties = schema["properties"]["model"]["properties"]

        for field_name in [
            "mla_latent_rank",
            "gqla_latent_rank",
            "gqla_num_groups",
            "gqla_decode_path",
            "mlra_latent_rank",
            "mlra_num_latent_heads",
            "tucker_query_rank",
            "tucker_key_rank",
            "tucker_value_rank",
            "iha_num_pseudo_heads",
            "gta_num_shared_groups",
            "gta_value_latent_rank",
            "mtla_latent_rank",
            "mtla_merge_factor",
            "mtla_stride",
            "msa_block_size",
            "msa_topk_blocks",
            "msa_index_dim",
            "msa_kl_loss_weight",
            "sparda_block_size",
            "sparda_topk_blocks",
            "sparda_forecast_dim",
        ]:
            self.assertIn(field_name, model_properties)


if __name__ == "__main__":
    unittest.main()
