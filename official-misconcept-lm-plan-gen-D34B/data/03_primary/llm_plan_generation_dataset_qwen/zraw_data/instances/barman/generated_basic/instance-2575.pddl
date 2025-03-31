(define (problem prob)
 (:domain barman)
 (:objects 
      shaker77 - shaker
      left right - hand
      shot340 - shot
      ingredient459 ingredient82 - ingredient
      cocktail495 - cocktail
      dispenser19 dispenser488 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker77)
  (ontable shot340)
  (dispenses dispenser19 ingredient459)
  (dispenses dispenser488 ingredient82)
  (clean shaker77)
  (clean shot340)
  (empty shaker77)
  (empty shot340)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker77 l0)
  (shaker-level shaker77 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail495 ingredient82)
  (cocktail-part2 cocktail495 ingredient459)
)
 (:goal
  (and
      (contains shot340 cocktail495)
)))
