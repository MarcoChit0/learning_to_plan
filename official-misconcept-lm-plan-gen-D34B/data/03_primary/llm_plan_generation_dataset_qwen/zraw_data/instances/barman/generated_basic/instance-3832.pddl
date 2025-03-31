(define (problem prob)
 (:domain barman)
 (:objects 
      shaker468 - shaker
      left right - hand
      shot362 shot82 shot86 - shot
      ingredient50 ingredient476 - ingredient
      cocktail492 - cocktail
      dispenser258 dispenser249 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker468)
  (ontable shot362)
  (ontable shot82)
  (ontable shot86)
  (dispenses dispenser258 ingredient50)
  (dispenses dispenser249 ingredient476)
  (clean shaker468)
  (clean shot362)
  (clean shot82)
  (clean shot86)
  (empty shaker468)
  (empty shot362)
  (empty shot82)
  (empty shot86)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker468 l0)
  (shaker-level shaker468 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail492 ingredient476)
  (cocktail-part2 cocktail492 ingredient50)
)
 (:goal
  (and
      (contains shot362 cocktail492)
      (contains shot82 ingredient50)
)))
