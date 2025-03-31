(define (problem prob)
 (:domain barman)
 (:objects 
      shaker446 - shaker
      left right - hand
      shot148 - shot
      ingredient282 ingredient231 ingredient334 ingredient288 - ingredient
      cocktail100 - cocktail
      dispenser122 dispenser362 dispenser72 dispenser131 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker446)
  (ontable shot148)
  (dispenses dispenser122 ingredient282)
  (dispenses dispenser362 ingredient231)
  (dispenses dispenser72 ingredient334)
  (dispenses dispenser131 ingredient288)
  (clean shaker446)
  (clean shot148)
  (empty shaker446)
  (empty shot148)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker446 l0)
  (shaker-level shaker446 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail100 ingredient288)
  (cocktail-part2 cocktail100 ingredient231)
)
 (:goal
  (and
      (contains shot148 cocktail100)
)))
