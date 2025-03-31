(define (problem prob)
 (:domain barman)
 (:objects 
      shaker37 - shaker
      left right - hand
      shot484 shot357 shot496 - shot
      ingredient277 ingredient221 - ingredient
      cocktail426 - cocktail
      dispenser433 dispenser188 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker37)
  (ontable shot484)
  (ontable shot357)
  (ontable shot496)
  (dispenses dispenser433 ingredient277)
  (dispenses dispenser188 ingredient221)
  (clean shaker37)
  (clean shot484)
  (clean shot357)
  (clean shot496)
  (empty shaker37)
  (empty shot484)
  (empty shot357)
  (empty shot496)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker37 l0)
  (shaker-level shaker37 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail426 ingredient221)
  (cocktail-part2 cocktail426 ingredient277)
)
 (:goal
  (and
      (contains shot484 cocktail426)
      (contains shot357 ingredient277)
)))
