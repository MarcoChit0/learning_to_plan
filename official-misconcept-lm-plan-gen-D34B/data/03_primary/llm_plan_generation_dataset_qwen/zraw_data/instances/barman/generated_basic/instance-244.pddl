(define (problem prob)
 (:domain barman)
 (:objects 
      shaker452 - shaker
      left right - hand
      shot174 - shot
      ingredient478 ingredient91 - ingredient
      cocktail1 - cocktail
      dispenser1 dispenser178 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker452)
  (ontable shot174)
  (dispenses dispenser1 ingredient478)
  (dispenses dispenser178 ingredient91)
  (clean shaker452)
  (clean shot174)
  (empty shaker452)
  (empty shot174)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker452 l0)
  (shaker-level shaker452 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient478)
  (cocktail-part2 cocktail1 ingredient91)
)
 (:goal
  (and
      (contains shot174 cocktail1)
)))
