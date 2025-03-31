(define (problem prob)
 (:domain barman)
 (:objects 
      shaker231 - shaker
      left right - hand
      shot108 shot86 - shot
      ingredient482 ingredient196 - ingredient
      cocktail390 - cocktail
      dispenser64 dispenser91 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker231)
  (ontable shot108)
  (ontable shot86)
  (dispenses dispenser64 ingredient482)
  (dispenses dispenser91 ingredient196)
  (clean shaker231)
  (clean shot108)
  (clean shot86)
  (empty shaker231)
  (empty shot108)
  (empty shot86)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker231 l0)
  (shaker-level shaker231 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail390 ingredient196)
  (cocktail-part2 cocktail390 ingredient482)
)
 (:goal
  (and
      (contains shot108 cocktail390)
)))
