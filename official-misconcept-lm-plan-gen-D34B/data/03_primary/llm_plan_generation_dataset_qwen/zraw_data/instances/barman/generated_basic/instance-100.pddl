(define (problem prob)
 (:domain barman)
 (:objects 
      shaker84 - shaker
      left right - hand
      shot497 - shot
      ingredient27 ingredient123 ingredient128 ingredient32 - ingredient
      cocktail1 - cocktail
      dispenser42 dispenser480 dispenser436 dispenser434 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker84)
  (ontable shot497)
  (dispenses dispenser42 ingredient27)
  (dispenses dispenser480 ingredient123)
  (dispenses dispenser436 ingredient128)
  (dispenses dispenser434 ingredient32)
  (clean shaker84)
  (clean shot497)
  (empty shaker84)
  (empty shot497)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker84 l0)
  (shaker-level shaker84 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient123)
  (cocktail-part2 cocktail1 ingredient27)
)
 (:goal
  (and
      (contains shot497 cocktail1)
)))
