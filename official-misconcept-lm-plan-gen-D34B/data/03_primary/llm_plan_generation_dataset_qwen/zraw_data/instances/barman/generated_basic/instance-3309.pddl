(define (problem prob)
 (:domain barman)
 (:objects 
      shaker37 - shaker
      left right - hand
      shot37 - shot
      ingredient317 ingredient91 - ingredient
      cocktail99 - cocktail
      dispenser226 dispenser296 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker37)
  (ontable shot37)
  (dispenses dispenser226 ingredient317)
  (dispenses dispenser296 ingredient91)
  (clean shaker37)
  (clean shot37)
  (empty shaker37)
  (empty shot37)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker37 l0)
  (shaker-level shaker37 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail99 ingredient91)
  (cocktail-part2 cocktail99 ingredient317)
)
 (:goal
  (and
      (contains shot37 cocktail99)
)))
