/// General dependencies
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use core::panic;
use tokenizers::tokenizer::Tokenizer;

/// phymes-core dependencies
use phymes_core::session::common_traits::{
    TensorProcessorTrait, TokenProcessorTrait, TokenWrapper, TokenizerConfig,
};

use super::candle_which::CandleModelWeights;

/// The actual asset struct
#[derive(Debug)]
pub struct CandleAsset {
    /// The device for computation
    pub device: Device,
    /// The loaded CandleModelInput
    pub model_weights: CandleModelWeights,
    /// The tokenizer
    pub tokenizer: Tokenizer,
    /// The tokenizer configurations
    pub tokenizer_config: TokenizerConfig,
    /// The default type to use
    pub dtype: DType,
}

impl CandleAsset {
    pub fn convert_vec_to_tensor(&self, input: &TokenWrapper) -> Result<Tensor> {
        match input {
            TokenWrapper::D1(tokens) => {
                let tensor = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
                Ok(tensor)
            }
            TokenWrapper::D2(tokens) => {
                let tmp: Vec<&[u32]> = tokens.iter().map(|x| x.as_slice()).collect();
                let tensor = Tensor::new(tmp, &self.device)?.contiguous()?;
                Ok(tensor)
            }
        }
    }
}

impl TokenProcessorTrait for CandleAsset {
    fn forward(
        &mut self,
        input: &TokenWrapper,
        index_position: usize,
        attention_mask: Option<&TokenWrapper>,
        include_lm_head: bool,
    ) -> Result<Tensor> {
        // Handle the input type
        let input = self.convert_vec_to_tensor(input)?;
        let attention_mask = match attention_mask {
            Some(m) => Some(self.convert_vec_to_tensor(m)?),
            None => None,
        };

        // Run forward inference
        let result: Tensor = match &mut self.model_weights {
            CandleModelWeights::Nomic(_) => panic!("Nomic is not yet implemented!"),
            CandleModelWeights::Bert(_) => panic!("Bert is not yet implemented!"),
            CandleModelWeights::QuantizedBert(_) => {
                panic!("Quantized Bert is not yet implemented!")
            }
            CandleModelWeights::QuantizedQwen2(mw) => {
                mw.forward(
                    &input,
                    index_position,
                    attention_mask.as_ref(),
                    include_lm_head,
                )?
                //logits.squeeze(0)? Needed for chat!
            }
            CandleModelWeights::QuantizedLlama(mw) => {
                mw.forward(&input, index_position)?
                //logits.squeeze(0)?
            }
        };
        Ok(result)
    }

    fn get_tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    fn get_tokenizer_config(&self) -> &TokenizerConfig {
        &self.tokenizer_config
    }
}

impl TensorProcessorTrait for CandleAsset {
    fn get_device(&self) -> &Device {
        &self.device
    }
}
