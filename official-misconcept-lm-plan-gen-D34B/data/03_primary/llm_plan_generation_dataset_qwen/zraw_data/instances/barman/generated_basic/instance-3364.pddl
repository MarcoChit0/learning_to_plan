(define (problem prob)
 (:domain barman)
 (:objects 
      shaker121 - shaker
      left right - hand
      shot391 - shot
      ingredient47 ingredient3 - ingredient
      cocktail115 - cocktail
      dispenser85 dispenser361 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker121)
  (ontable shot391)
  (dispenses dispenser85 ingredient47)
  (dispenses dispenser361 ingredient3)
  (clean shaker121)
  (clean shot391)
  (empty shaker121)
  (empty shot391)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker121 l0)
  (shaker-level shaker121 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail115 ingredient3)
  (cocktail-part2 cocktail115 ingredient47)
)
 (:goal
  (and
      (contains shot391 cocktail115)
)))
