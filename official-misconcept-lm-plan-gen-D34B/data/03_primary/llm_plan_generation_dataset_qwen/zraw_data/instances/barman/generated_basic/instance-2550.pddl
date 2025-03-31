(define (problem prob)
 (:domain barman)
 (:objects 
      shaker450 - shaker
      left right - hand
      shot82 shot150 - shot
      ingredient318 ingredient245 - ingredient
      cocktail284 - cocktail
      dispenser489 dispenser340 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker450)
  (ontable shot82)
  (ontable shot150)
  (dispenses dispenser489 ingredient318)
  (dispenses dispenser340 ingredient245)
  (clean shaker450)
  (clean shot82)
  (clean shot150)
  (empty shaker450)
  (empty shot82)
  (empty shot150)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker450 l0)
  (shaker-level shaker450 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail284 ingredient245)
  (cocktail-part2 cocktail284 ingredient318)
)
 (:goal
  (and
      (contains shot82 cocktail284)
)))
