(define (problem prob)
 (:domain barman)
 (:objects 
      shaker198 - shaker
      left right - hand
      shot437 - shot
      ingredient1 ingredient9 ingredient88 ingredient53 - ingredient
      cocktail108 - cocktail
      dispenser299 dispenser244 dispenser486 dispenser71 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker198)
  (ontable shot437)
  (dispenses dispenser299 ingredient1)
  (dispenses dispenser244 ingredient9)
  (dispenses dispenser486 ingredient88)
  (dispenses dispenser71 ingredient53)
  (clean shaker198)
  (clean shot437)
  (empty shaker198)
  (empty shot437)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker198 l0)
  (shaker-level shaker198 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail108 ingredient88)
  (cocktail-part2 cocktail108 ingredient9)
)
 (:goal
  (and
      (contains shot437 cocktail108)
)))
