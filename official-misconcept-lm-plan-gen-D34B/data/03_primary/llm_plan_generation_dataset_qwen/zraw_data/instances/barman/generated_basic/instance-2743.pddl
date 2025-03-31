(define (problem prob)
 (:domain barman)
 (:objects 
      shaker468 - shaker
      left right - hand
      shot265 - shot
      ingredient170 ingredient69 - ingredient
      cocktail482 - cocktail
      dispenser18 dispenser222 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker468)
  (ontable shot265)
  (dispenses dispenser18 ingredient170)
  (dispenses dispenser222 ingredient69)
  (clean shaker468)
  (clean shot265)
  (empty shaker468)
  (empty shot265)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker468 l0)
  (shaker-level shaker468 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail482 ingredient69)
  (cocktail-part2 cocktail482 ingredient170)
)
 (:goal
  (and
      (contains shot265 cocktail482)
)))
