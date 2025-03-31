(define (problem prob)
 (:domain barman)
 (:objects 
      shaker101 - shaker
      left right - hand
      shot417 - shot
      ingredient472 ingredient453 - ingredient
      cocktail1 - cocktail
      dispenser69 dispenser461 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker101)
  (ontable shot417)
  (dispenses dispenser69 ingredient472)
  (dispenses dispenser461 ingredient453)
  (clean shaker101)
  (clean shot417)
  (empty shaker101)
  (empty shot417)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker101 l0)
  (shaker-level shaker101 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient472)
  (cocktail-part2 cocktail1 ingredient453)
)
 (:goal
  (and
      (contains shot417 cocktail1)
)))
