(define (problem prob)
 (:domain barman)
 (:objects 
      shaker25 - shaker
      left right - hand
      shot415 - shot
      ingredient382 ingredient393 - ingredient
      cocktail399 - cocktail
      dispenser244 dispenser243 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker25)
  (ontable shot415)
  (dispenses dispenser244 ingredient382)
  (dispenses dispenser243 ingredient393)
  (clean shaker25)
  (clean shot415)
  (empty shaker25)
  (empty shot415)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker25 l0)
  (shaker-level shaker25 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail399 ingredient382)
  (cocktail-part2 cocktail399 ingredient393)
)
 (:goal
  (and
      (contains shot415 cocktail399)
)))
