(define (problem prob)
 (:domain barman)
 (:objects 
      shaker23 - shaker
      left right - hand
      shot10 shot450 - shot
      ingredient81 ingredient143 ingredient179 - ingredient
      cocktail363 - cocktail
      dispenser348 dispenser122 dispenser131 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker23)
  (ontable shot10)
  (ontable shot450)
  (dispenses dispenser348 ingredient81)
  (dispenses dispenser122 ingredient143)
  (dispenses dispenser131 ingredient179)
  (clean shaker23)
  (clean shot10)
  (clean shot450)
  (empty shaker23)
  (empty shot10)
  (empty shot450)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker23 l0)
  (shaker-level shaker23 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail363 ingredient143)
  (cocktail-part2 cocktail363 ingredient179)
)
 (:goal
  (and
      (contains shot10 cocktail363)
)))
