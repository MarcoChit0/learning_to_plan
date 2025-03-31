(define (problem prob)
 (:domain barman)
 (:objects 
      shaker260 - shaker
      left right - hand
      shot37 - shot
      ingredient74 ingredient360 - ingredient
      cocktail56 - cocktail
      dispenser404 dispenser381 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker260)
  (ontable shot37)
  (dispenses dispenser404 ingredient74)
  (dispenses dispenser381 ingredient360)
  (clean shaker260)
  (clean shot37)
  (empty shaker260)
  (empty shot37)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker260 l0)
  (shaker-level shaker260 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail56 ingredient360)
  (cocktail-part2 cocktail56 ingredient74)
)
 (:goal
  (and
      (contains shot37 cocktail56)
)))
