(define (problem prob)
 (:domain barman)
 (:objects 
      shaker187 - shaker
      left right - hand
      shot149 - shot
      ingredient155 ingredient262 ingredient121 ingredient368 - ingredient
      cocktail70 - cocktail
      dispenser345 dispenser263 dispenser190 dispenser410 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker187)
  (ontable shot149)
  (dispenses dispenser345 ingredient155)
  (dispenses dispenser263 ingredient262)
  (dispenses dispenser190 ingredient121)
  (dispenses dispenser410 ingredient368)
  (clean shaker187)
  (clean shot149)
  (empty shaker187)
  (empty shot149)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker187 l0)
  (shaker-level shaker187 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail70 ingredient368)
  (cocktail-part2 cocktail70 ingredient121)
)
 (:goal
  (and
      (contains shot149 cocktail70)
)))
