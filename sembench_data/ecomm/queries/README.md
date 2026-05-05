should we include tabular data in basic queries? 
Gautham mentioned queries should not be evaluable with classical SQL, but with tabular data, this would always be the case, no?

Semantic ranking is difficult for this dataset because there are hardly any non-categorical columns. price, discountedPrice, rating, weight come to my mind, but this is hard to infer (probably).
maybe discount text and then infer discounted price? or is this just sem_map?

semantic aggregation will also be hard to do.
maybe: pre-fitler and then ask to "give an overview over the products beind sold. first summarize ..." and then do a vector comparison on that?? maybe vector comparison is not enough because it might disregard hard facts? what would be a nice use case even?

