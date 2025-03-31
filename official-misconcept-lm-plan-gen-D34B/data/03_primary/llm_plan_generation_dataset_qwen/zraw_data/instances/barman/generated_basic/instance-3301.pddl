(define (problem prob)
 (:domain barman)
 (:objects 
      shaker452 - shaker
      left right - hand
      shot182 - shot
      ingredient421 ingredient288 - ingredient
      cocktail310 - cocktail
      dispenser109 dispenser198 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker452)
  (ontable shot182)
  (dispenses dispenser109 ingredient421)
  (dispenses dispenser198 ingredient288)
  (clean shaker452)
  (clean shot182)
  (empty shaker452)
  (empty shot182)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker452 l0)
  (shaker-level shaker452 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail310 ingredient288)
  (cocktail-part2 cocktail310 ingredient421)
)
 (:goal
  (and
      (contains shot182 cocktail310)
)))
