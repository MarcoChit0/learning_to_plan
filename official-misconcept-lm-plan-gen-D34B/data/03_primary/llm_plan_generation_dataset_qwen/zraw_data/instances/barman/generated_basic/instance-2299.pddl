(define (problem prob)
 (:domain barman)
 (:objects 
      shaker316 - shaker
      left right - hand
      shot140 - shot
      ingredient309 ingredient201 ingredient224 - ingredient
      cocktail346 - cocktail
      dispenser363 dispenser275 dispenser213 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker316)
  (ontable shot140)
  (dispenses dispenser363 ingredient309)
  (dispenses dispenser275 ingredient201)
  (dispenses dispenser213 ingredient224)
  (clean shaker316)
  (clean shot140)
  (empty shaker316)
  (empty shot140)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker316 l0)
  (shaker-level shaker316 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail346 ingredient201)
  (cocktail-part2 cocktail346 ingredient309)
)
 (:goal
  (and
      (contains shot140 cocktail346)
)))
