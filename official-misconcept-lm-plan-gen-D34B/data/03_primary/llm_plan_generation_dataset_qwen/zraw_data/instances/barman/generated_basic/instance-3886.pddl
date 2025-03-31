(define (problem prob)
 (:domain barman)
 (:objects 
      shaker485 - shaker
      left right - hand
      shot464 shot304 - shot
      ingredient6 ingredient17 ingredient360 - ingredient
      cocktail229 - cocktail
      dispenser470 dispenser153 dispenser458 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker485)
  (ontable shot464)
  (ontable shot304)
  (dispenses dispenser470 ingredient6)
  (dispenses dispenser153 ingredient17)
  (dispenses dispenser458 ingredient360)
  (clean shaker485)
  (clean shot464)
  (clean shot304)
  (empty shaker485)
  (empty shot464)
  (empty shot304)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker485 l0)
  (shaker-level shaker485 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail229 ingredient360)
  (cocktail-part2 cocktail229 ingredient6)
)
 (:goal
  (and
      (contains shot464 cocktail229)
)))
