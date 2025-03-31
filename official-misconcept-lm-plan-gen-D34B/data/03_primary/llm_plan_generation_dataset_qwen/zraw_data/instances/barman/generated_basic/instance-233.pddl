(define (problem prob)
 (:domain barman)
 (:objects 
      shaker41 - shaker
      left right - hand
      shot352 - shot
      ingredient431 ingredient478 - ingredient
      cocktail1 - cocktail
      dispenser69 dispenser7 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker41)
  (ontable shot352)
  (dispenses dispenser69 ingredient431)
  (dispenses dispenser7 ingredient478)
  (clean shaker41)
  (clean shot352)
  (empty shaker41)
  (empty shot352)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker41 l0)
  (shaker-level shaker41 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient478)
  (cocktail-part2 cocktail1 ingredient431)
)
 (:goal
  (and
      (contains shot352 cocktail1)
)))
