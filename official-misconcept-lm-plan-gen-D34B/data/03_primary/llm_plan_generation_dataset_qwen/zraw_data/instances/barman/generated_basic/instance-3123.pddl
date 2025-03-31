(define (problem prob)
 (:domain barman)
 (:objects 
      shaker89 - shaker
      left right - hand
      shot86 shot88 - shot
      ingredient376 ingredient486 - ingredient
      cocktail55 - cocktail
      dispenser419 dispenser266 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker89)
  (ontable shot86)
  (ontable shot88)
  (dispenses dispenser419 ingredient376)
  (dispenses dispenser266 ingredient486)
  (clean shaker89)
  (clean shot86)
  (clean shot88)
  (empty shaker89)
  (empty shot86)
  (empty shot88)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker89 l0)
  (shaker-level shaker89 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail55 ingredient486)
  (cocktail-part2 cocktail55 ingredient376)
)
 (:goal
  (and
      (contains shot86 cocktail55)
)))
