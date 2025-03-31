(define (problem prob)
 (:domain barman)
 (:objects 
      shaker383 - shaker
      left right - hand
      shot273 shot9 - shot
      ingredient101 ingredient360 ingredient231 - ingredient
      cocktail284 - cocktail
      dispenser226 dispenser437 dispenser223 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker383)
  (ontable shot273)
  (ontable shot9)
  (dispenses dispenser226 ingredient101)
  (dispenses dispenser437 ingredient360)
  (dispenses dispenser223 ingredient231)
  (clean shaker383)
  (clean shot273)
  (clean shot9)
  (empty shaker383)
  (empty shot273)
  (empty shot9)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker383 l0)
  (shaker-level shaker383 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail284 ingredient231)
  (cocktail-part2 cocktail284 ingredient360)
)
 (:goal
  (and
      (contains shot273 cocktail284)
)))
