(define (problem prob)
 (:domain barman)
 (:objects 
      shaker59 - shaker
      left right - hand
      shot116 - shot
      ingredient6 ingredient106 ingredient57 - ingredient
      cocktail210 - cocktail
      dispenser442 dispenser432 dispenser134 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker59)
  (ontable shot116)
  (dispenses dispenser442 ingredient6)
  (dispenses dispenser432 ingredient106)
  (dispenses dispenser134 ingredient57)
  (clean shaker59)
  (clean shot116)
  (empty shaker59)
  (empty shot116)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker59 l0)
  (shaker-level shaker59 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail210 ingredient106)
  (cocktail-part2 cocktail210 ingredient6)
)
 (:goal
  (and
      (contains shot116 cocktail210)
)))
