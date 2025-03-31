(define (problem prob)
 (:domain barman)
 (:objects 
      shaker121 - shaker
      left right - hand
      shot4 shot101 - shot
      ingredient56 ingredient292 - ingredient
      cocktail222 - cocktail
      dispenser487 dispenser286 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker121)
  (ontable shot4)
  (ontable shot101)
  (dispenses dispenser487 ingredient56)
  (dispenses dispenser286 ingredient292)
  (clean shaker121)
  (clean shot4)
  (clean shot101)
  (empty shaker121)
  (empty shot4)
  (empty shot101)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker121 l0)
  (shaker-level shaker121 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail222 ingredient292)
  (cocktail-part2 cocktail222 ingredient56)
)
 (:goal
  (and
      (contains shot4 cocktail222)
)))
