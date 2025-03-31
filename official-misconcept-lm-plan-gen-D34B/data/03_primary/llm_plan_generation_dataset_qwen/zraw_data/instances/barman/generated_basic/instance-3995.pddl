(define (problem prob)
 (:domain barman)
 (:objects 
      shaker322 - shaker
      left right - hand
      shot442 shot362 shot162 - shot
      ingredient418 ingredient352 - ingredient
      cocktail140 - cocktail
      dispenser226 dispenser448 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker322)
  (ontable shot442)
  (ontable shot362)
  (ontable shot162)
  (dispenses dispenser226 ingredient418)
  (dispenses dispenser448 ingredient352)
  (clean shaker322)
  (clean shot442)
  (clean shot362)
  (clean shot162)
  (empty shaker322)
  (empty shot442)
  (empty shot362)
  (empty shot162)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker322 l0)
  (shaker-level shaker322 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail140 ingredient352)
  (cocktail-part2 cocktail140 ingredient418)
)
 (:goal
  (and
      (contains shot442 cocktail140)
      (contains shot362 ingredient418)
)))
