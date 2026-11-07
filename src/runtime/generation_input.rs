#[cfg(feature = "isf")]
pub(crate) enum GenerationInput<'a> {
    Text(&'a str),
    TokenIds(&'a [u32]),
}

#[cfg(feature = "isf")]
pub(crate) fn validate_prompt_ids(ids: &[u32], vocab: usize, context: usize) -> Result<(), String> {
    if ids.is_empty() {
        return Err("invalid_prompt_tokens: empty prompt".into());
    }
    if ids.len() > context {
        return Err("invalid_prompt_tokens: prompt exceeds context".into());
    }
    if ids.iter().any(|&id| id as usize >= vocab) {
        return Err("invalid_prompt_tokens: token outside vocabulary".into());
    }
    Ok(())
}

#[cfg(all(test, feature = "isf"))]
mod tests {
    use super::validate_prompt_ids;
    #[test]
    fn injected_tokens_enforce_vocab_and_context_before_dispatch() {
        assert!(validate_prompt_ids(&[0, 7], 8, 2).is_ok());
        assert!(validate_prompt_ids(&[], 8, 2).is_err());
        assert!(validate_prompt_ids(&[8], 8, 2).is_err());
        assert!(validate_prompt_ids(&[0, 1, 2], 8, 2).is_err());
    }
}
