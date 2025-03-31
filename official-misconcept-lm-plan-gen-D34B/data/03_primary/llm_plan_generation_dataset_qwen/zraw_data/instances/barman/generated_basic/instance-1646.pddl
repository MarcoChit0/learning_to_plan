(define (problem prob)
 (:domain barman)
 (:objects 
      shaker167 - shaker
      left right - hand
      shot368 shot79 shot358 - shot
      ingredient50 ingredient464 ingredient120 ingredient425 - ingredient
      cocktail261 - cocktail
      dispenser297 dispenser390 dispenser201 dispenser400 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker167)
  (ontable shot368)
  (ontable shot79)
  (ontable shot358)
  (dispenses dispenser297 ingredient50)
  (dispenses dispenser390 ingredient464)
  (dispenses dispenser201 ingredient120)
  (dispenses dispenser400 ingredient425)
  (clean shaker167)
  (clean shot368)
  (clean shot79)
  (clean shot358)
  (empty shaker167)
  (empty shot368)
  (empty shot79)
  (empty shot358)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker167 l0)
  (shaker-level shaker167 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail261 ingredient464)
  (cocktail-part2 cocktail261 ingredient120)
)
 (:goal
  (and
      (contains shot368 cocktail261)
      (contains shot79 cocktail261)
)))
