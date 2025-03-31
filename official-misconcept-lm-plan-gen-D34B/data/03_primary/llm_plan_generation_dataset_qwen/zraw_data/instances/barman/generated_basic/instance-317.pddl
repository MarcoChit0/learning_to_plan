(define (problem prob)
 (:domain barman)
 (:objects 
      shaker78 - shaker
      left right - hand
      shot54 shot268 - shot
      ingredient215 ingredient208 - ingredient
      cocktail1 - cocktail
      dispenser270 dispenser147 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker78)
  (ontable shot54)
  (ontable shot268)
  (dispenses dispenser270 ingredient215)
  (dispenses dispenser147 ingredient208)
  (clean shaker78)
  (clean shot54)
  (clean shot268)
  (empty shaker78)
  (empty shot54)
  (empty shot268)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker78 l0)
  (shaker-level shaker78 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient215)
  (cocktail-part2 cocktail1 ingredient208)
)
 (:goal
  (and
      (contains shot54 cocktail1)
)))
