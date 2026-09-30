import torch

from axosim_demo.whole_brain import StreamingCrossingDecoder


def test_streaming_crossing_decoder_preserves_block_boundaries():
    decoder = StreamingCrossingDecoder(3, threshold=0.0, device='cpu')
    first = torch.tensor([
        [-1.0, 0.2, 0.3, -0.1],
        [0.2, 0.3, -0.2, 0.1],
        [-1.0, -0.5, -0.2, -0.1],
    ])
    second = torch.tensor([
        [0.4, -0.2, 0.1, 0.2],
        [0.2, -0.1, -0.2, -0.3],
        [0.1, 0.2, -0.1, 0.0],
    ])
    torch.testing.assert_close(decoder(first), torch.tensor([
        [False, True, False, False],
        [True, False, False, True],
        [False, False, False, False],
    ]))
    torch.testing.assert_close(decoder(second), torch.tensor([
        [True, False, True, False],
        [False, False, False, False],
        [True, False, False, True],
    ]))


def test_decoder_can_be_reset_without_reusing_prior_activity():
    decoder = StreamingCrossingDecoder(1, threshold=0.0, device='cpu')
    decoder(torch.tensor([[1.0, 1.0, 1.0, 1.0]]))
    decoder.reset()
    torch.testing.assert_close(
        decoder(torch.tensor([[1.0, 1.0, 1.0, 1.0]])),
        torch.tensor([[True, False, False, False]]),
    )
