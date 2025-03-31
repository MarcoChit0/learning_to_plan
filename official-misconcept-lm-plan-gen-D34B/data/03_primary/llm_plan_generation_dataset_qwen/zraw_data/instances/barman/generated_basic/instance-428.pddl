(define (problem prob)
 (:domain barman)
 (:objects 
      shaker406 - shaker
      left right - hand
      shot329 - shot
      ingredient72 ingredient462 - ingredient
      cocktail1 - cocktail
      dispenser132 dispenser356 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker406)
  (ontable shot329)
  (dispenses dispenser132 ingredient72)
  (dispenses dispenser356 ingredient462)
  (clean shaker406)
  (clean shot329)
  (empty shaker406)
  (empty shot329)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker406 l0)
  (shaker-level shaker406 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient462)
  (cocktail-part2 cocktail1 ingredient72)
)
 (:goal
  (and
      (contains shot329 cocktail1)
)))
