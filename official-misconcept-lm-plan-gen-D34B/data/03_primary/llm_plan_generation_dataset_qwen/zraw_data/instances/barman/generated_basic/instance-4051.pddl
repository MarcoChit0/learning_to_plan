(define (problem prob)
 (:domain barman)
 (:objects 
      shaker308 - shaker
      left right - hand
      shot326 shot379 - shot
      ingredient421 ingredient159 ingredient437 ingredient309 - ingredient
      cocktail74 - cocktail
      dispenser229 dispenser246 dispenser81 dispenser322 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker308)
  (ontable shot326)
  (ontable shot379)
  (dispenses dispenser229 ingredient421)
  (dispenses dispenser246 ingredient159)
  (dispenses dispenser81 ingredient437)
  (dispenses dispenser322 ingredient309)
  (clean shaker308)
  (clean shot326)
  (clean shot379)
  (empty shaker308)
  (empty shot326)
  (empty shot379)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker308 l0)
  (shaker-level shaker308 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail74 ingredient159)
  (cocktail-part2 cocktail74 ingredient309)
)
 (:goal
  (and
      (contains shot326 cocktail74)
)))
