(define (problem prob)
 (:domain barman)
 (:objects 
      shaker328 - shaker
      left right - hand
      shot331 shot86 - shot
      ingredient85 ingredient47 ingredient205 - ingredient
      cocktail434 - cocktail
      dispenser346 dispenser160 dispenser462 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker328)
  (ontable shot331)
  (ontable shot86)
  (dispenses dispenser346 ingredient85)
  (dispenses dispenser160 ingredient47)
  (dispenses dispenser462 ingredient205)
  (clean shaker328)
  (clean shot331)
  (clean shot86)
  (empty shaker328)
  (empty shot331)
  (empty shot86)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker328 l0)
  (shaker-level shaker328 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail434 ingredient85)
  (cocktail-part2 cocktail434 ingredient47)
)
 (:goal
  (and
      (contains shot331 cocktail434)
)))
