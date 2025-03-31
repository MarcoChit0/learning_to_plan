(define (problem prob)
 (:domain barman)
 (:objects 
      shaker110 - shaker
      left right - hand
      shot355 shot262 - shot
      ingredient112 ingredient123 - ingredient
      cocktail469 - cocktail
      dispenser401 dispenser152 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker110)
  (ontable shot355)
  (ontable shot262)
  (dispenses dispenser401 ingredient112)
  (dispenses dispenser152 ingredient123)
  (clean shaker110)
  (clean shot355)
  (clean shot262)
  (empty shaker110)
  (empty shot355)
  (empty shot262)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker110 l0)
  (shaker-level shaker110 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail469 ingredient123)
  (cocktail-part2 cocktail469 ingredient112)
)
 (:goal
  (and
      (contains shot355 cocktail469)
)))
