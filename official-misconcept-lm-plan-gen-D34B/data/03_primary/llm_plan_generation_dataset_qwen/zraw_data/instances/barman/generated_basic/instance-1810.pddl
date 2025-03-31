(define (problem prob)
 (:domain barman)
 (:objects 
      shaker401 - shaker
      left right - hand
      shot79 shot470 - shot
      ingredient418 ingredient7 ingredient373 ingredient382 - ingredient
      cocktail134 - cocktail
      dispenser299 dispenser406 dispenser39 dispenser489 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker401)
  (ontable shot79)
  (ontable shot470)
  (dispenses dispenser299 ingredient418)
  (dispenses dispenser406 ingredient7)
  (dispenses dispenser39 ingredient373)
  (dispenses dispenser489 ingredient382)
  (clean shaker401)
  (clean shot79)
  (clean shot470)
  (empty shaker401)
  (empty shot79)
  (empty shot470)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker401 l0)
  (shaker-level shaker401 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail134 ingredient418)
  (cocktail-part2 cocktail134 ingredient382)
)
 (:goal
  (and
      (contains shot79 cocktail134)
)))
