(define (problem prob)
 (:domain barman)
 (:objects 
      shaker29 - shaker
      left right - hand
      shot450 - shot
      ingredient206 ingredient79 - ingredient
      cocktail208 - cocktail
      dispenser421 dispenser9 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker29)
  (ontable shot450)
  (dispenses dispenser421 ingredient206)
  (dispenses dispenser9 ingredient79)
  (clean shaker29)
  (clean shot450)
  (empty shaker29)
  (empty shot450)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker29 l0)
  (shaker-level shaker29 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail208 ingredient206)
  (cocktail-part2 cocktail208 ingredient79)
)
 (:goal
  (and
      (contains shot450 cocktail208)
)))
