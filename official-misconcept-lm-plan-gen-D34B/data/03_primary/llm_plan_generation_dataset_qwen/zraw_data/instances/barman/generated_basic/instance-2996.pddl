(define (problem prob)
 (:domain barman)
 (:objects 
      shaker258 - shaker
      left right - hand
      shot185 shot180 - shot
      ingredient225 ingredient451 ingredient80 ingredient101 - ingredient
      cocktail48 - cocktail
      dispenser223 dispenser123 dispenser162 dispenser450 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker258)
  (ontable shot185)
  (ontable shot180)
  (dispenses dispenser223 ingredient225)
  (dispenses dispenser123 ingredient451)
  (dispenses dispenser162 ingredient80)
  (dispenses dispenser450 ingredient101)
  (clean shaker258)
  (clean shot185)
  (clean shot180)
  (empty shaker258)
  (empty shot185)
  (empty shot180)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker258 l0)
  (shaker-level shaker258 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail48 ingredient80)
  (cocktail-part2 cocktail48 ingredient101)
)
 (:goal
  (and
      (contains shot185 cocktail48)
)))
