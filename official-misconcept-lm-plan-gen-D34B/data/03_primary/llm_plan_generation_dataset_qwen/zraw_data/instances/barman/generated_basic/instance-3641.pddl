(define (problem prob)
 (:domain barman)
 (:objects 
      shaker63 - shaker
      left right - hand
      shot83 - shot
      ingredient414 ingredient24 ingredient381 ingredient251 - ingredient
      cocktail255 - cocktail
      dispenser228 dispenser259 dispenser304 dispenser297 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker63)
  (ontable shot83)
  (dispenses dispenser228 ingredient414)
  (dispenses dispenser259 ingredient24)
  (dispenses dispenser304 ingredient381)
  (dispenses dispenser297 ingredient251)
  (clean shaker63)
  (clean shot83)
  (empty shaker63)
  (empty shot83)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker63 l0)
  (shaker-level shaker63 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail255 ingredient251)
  (cocktail-part2 cocktail255 ingredient414)
)
 (:goal
  (and
      (contains shot83 cocktail255)
)))
