(define (problem prob)
 (:domain barman)
 (:objects 
      shaker309 - shaker
      left right - hand
      shot381 shot17 shot156 - shot
      ingredient399 ingredient339 - ingredient
      cocktail475 - cocktail
      dispenser130 dispenser442 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker309)
  (ontable shot381)
  (ontable shot17)
  (ontable shot156)
  (dispenses dispenser130 ingredient399)
  (dispenses dispenser442 ingredient339)
  (clean shaker309)
  (clean shot381)
  (clean shot17)
  (clean shot156)
  (empty shaker309)
  (empty shot381)
  (empty shot17)
  (empty shot156)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker309 l0)
  (shaker-level shaker309 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail475 ingredient339)
  (cocktail-part2 cocktail475 ingredient399)
)
 (:goal
  (and
      (contains shot381 cocktail475)
      (contains shot17 ingredient339)
)))
