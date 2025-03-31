(define (problem prob)
 (:domain barman)
 (:objects 
      shaker291 - shaker
      left right - hand
      shot462 - shot
      ingredient77 ingredient57 - ingredient
      cocktail243 - cocktail
      dispenser297 dispenser345 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker291)
  (ontable shot462)
  (dispenses dispenser297 ingredient77)
  (dispenses dispenser345 ingredient57)
  (clean shaker291)
  (clean shot462)
  (empty shaker291)
  (empty shot462)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker291 l0)
  (shaker-level shaker291 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail243 ingredient57)
  (cocktail-part2 cocktail243 ingredient77)
)
 (:goal
  (and
      (contains shot462 cocktail243)
)))
