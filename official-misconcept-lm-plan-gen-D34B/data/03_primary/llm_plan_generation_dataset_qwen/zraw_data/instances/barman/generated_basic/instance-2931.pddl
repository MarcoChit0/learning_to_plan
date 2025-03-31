(define (problem prob)
 (:domain barman)
 (:objects 
      shaker390 - shaker
      left right - hand
      shot191 - shot
      ingredient42 ingredient69 - ingredient
      cocktail49 - cocktail
      dispenser421 dispenser382 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker390)
  (ontable shot191)
  (dispenses dispenser421 ingredient42)
  (dispenses dispenser382 ingredient69)
  (clean shaker390)
  (clean shot191)
  (empty shaker390)
  (empty shot191)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker390 l0)
  (shaker-level shaker390 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail49 ingredient42)
  (cocktail-part2 cocktail49 ingredient69)
)
 (:goal
  (and
      (contains shot191 cocktail49)
)))
