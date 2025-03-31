(define (problem prob)
 (:domain barman)
 (:objects 
      shaker67 - shaker
      left right - hand
      shot478 shot422 - shot
      ingredient205 ingredient320 - ingredient
      cocktail320 - cocktail
      dispenser215 dispenser395 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker67)
  (ontable shot478)
  (ontable shot422)
  (dispenses dispenser215 ingredient205)
  (dispenses dispenser395 ingredient320)
  (clean shaker67)
  (clean shot478)
  (clean shot422)
  (empty shaker67)
  (empty shot478)
  (empty shot422)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker67 l0)
  (shaker-level shaker67 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail320 ingredient320)
  (cocktail-part2 cocktail320 ingredient205)
)
 (:goal
  (and
      (contains shot478 cocktail320)
)))
