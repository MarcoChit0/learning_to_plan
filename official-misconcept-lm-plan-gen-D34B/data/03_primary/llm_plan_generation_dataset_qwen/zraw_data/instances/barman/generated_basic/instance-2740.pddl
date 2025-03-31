(define (problem prob)
 (:domain barman)
 (:objects 
      shaker396 - shaker
      left right - hand
      shot457 - shot
      ingredient433 ingredient338 - ingredient
      cocktail371 - cocktail
      dispenser16 dispenser80 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker396)
  (ontable shot457)
  (dispenses dispenser16 ingredient433)
  (dispenses dispenser80 ingredient338)
  (clean shaker396)
  (clean shot457)
  (empty shaker396)
  (empty shot457)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker396 l0)
  (shaker-level shaker396 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail371 ingredient338)
  (cocktail-part2 cocktail371 ingredient433)
)
 (:goal
  (and
      (contains shot457 cocktail371)
)))
