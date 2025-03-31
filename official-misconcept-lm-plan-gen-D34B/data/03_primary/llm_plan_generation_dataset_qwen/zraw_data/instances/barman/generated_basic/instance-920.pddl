(define (problem prob)
 (:domain barman)
 (:objects 
      shaker75 - shaker
      left right - hand
      shot49 shot54 - shot
      ingredient441 ingredient80 ingredient156 - ingredient
      cocktail1 - cocktail
      dispenser151 dispenser495 dispenser379 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker75)
  (ontable shot49)
  (ontable shot54)
  (dispenses dispenser151 ingredient441)
  (dispenses dispenser495 ingredient80)
  (dispenses dispenser379 ingredient156)
  (clean shaker75)
  (clean shot49)
  (clean shot54)
  (empty shaker75)
  (empty shot49)
  (empty shot54)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker75 l0)
  (shaker-level shaker75 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient80)
  (cocktail-part2 cocktail1 ingredient156)
)
 (:goal
  (and
      (contains shot49 cocktail1)
)))
