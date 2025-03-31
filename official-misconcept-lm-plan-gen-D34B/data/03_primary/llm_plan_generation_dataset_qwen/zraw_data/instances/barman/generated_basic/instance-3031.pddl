(define (problem prob)
 (:domain barman)
 (:objects 
      shaker78 - shaker
      left right - hand
      shot99 - shot
      ingredient80 ingredient213 ingredient142 - ingredient
      cocktail371 - cocktail
      dispenser494 dispenser416 dispenser70 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker78)
  (ontable shot99)
  (dispenses dispenser494 ingredient80)
  (dispenses dispenser416 ingredient213)
  (dispenses dispenser70 ingredient142)
  (clean shaker78)
  (clean shot99)
  (empty shaker78)
  (empty shot99)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker78 l0)
  (shaker-level shaker78 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail371 ingredient80)
  (cocktail-part2 cocktail371 ingredient142)
)
 (:goal
  (and
      (contains shot99 cocktail371)
)))
