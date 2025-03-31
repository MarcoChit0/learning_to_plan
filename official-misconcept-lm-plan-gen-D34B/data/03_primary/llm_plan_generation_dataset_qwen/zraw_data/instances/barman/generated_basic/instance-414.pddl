(define (problem prob)
 (:domain barman)
 (:objects 
      shaker472 - shaker
      left right - hand
      shot249 - shot
      ingredient450 ingredient460 - ingredient
      cocktail1 - cocktail
      dispenser472 dispenser196 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker472)
  (ontable shot249)
  (dispenses dispenser472 ingredient450)
  (dispenses dispenser196 ingredient460)
  (clean shaker472)
  (clean shot249)
  (empty shaker472)
  (empty shot249)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker472 l0)
  (shaker-level shaker472 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient450)
  (cocktail-part2 cocktail1 ingredient460)
)
 (:goal
  (and
      (contains shot249 cocktail1)
)))
