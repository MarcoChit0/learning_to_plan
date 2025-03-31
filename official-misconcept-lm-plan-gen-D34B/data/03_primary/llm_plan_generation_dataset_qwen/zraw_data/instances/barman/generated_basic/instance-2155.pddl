(define (problem prob)
 (:domain barman)
 (:objects 
      shaker186 - shaker
      left right - hand
      shot342 - shot
      ingredient338 ingredient76 - ingredient
      cocktail497 - cocktail
      dispenser98 dispenser430 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker186)
  (ontable shot342)
  (dispenses dispenser98 ingredient338)
  (dispenses dispenser430 ingredient76)
  (clean shaker186)
  (clean shot342)
  (empty shaker186)
  (empty shot342)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker186 l0)
  (shaker-level shaker186 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail497 ingredient338)
  (cocktail-part2 cocktail497 ingredient76)
)
 (:goal
  (and
      (contains shot342 cocktail497)
)))
