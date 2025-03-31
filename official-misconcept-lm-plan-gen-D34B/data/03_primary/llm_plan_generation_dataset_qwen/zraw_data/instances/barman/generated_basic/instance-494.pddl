(define (problem prob)
 (:domain barman)
 (:objects 
      shaker367 - shaker
      left right - hand
      shot187 shot168 - shot
      ingredient183 ingredient306 ingredient159 ingredient492 - ingredient
      cocktail1 - cocktail
      dispenser388 dispenser444 dispenser486 dispenser440 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker367)
  (ontable shot187)
  (ontable shot168)
  (dispenses dispenser388 ingredient183)
  (dispenses dispenser444 ingredient306)
  (dispenses dispenser486 ingredient159)
  (dispenses dispenser440 ingredient492)
  (clean shaker367)
  (clean shot187)
  (clean shot168)
  (empty shaker367)
  (empty shot187)
  (empty shot168)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker367 l0)
  (shaker-level shaker367 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient159)
  (cocktail-part2 cocktail1 ingredient492)
)
 (:goal
  (and
      (contains shot187 cocktail1)
)))
