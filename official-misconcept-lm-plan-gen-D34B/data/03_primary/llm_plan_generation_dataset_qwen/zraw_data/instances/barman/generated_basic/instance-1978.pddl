(define (problem prob)
 (:domain barman)
 (:objects 
      shaker486 - shaker
      left right - hand
      shot7 shot212 - shot
      ingredient489 ingredient38 ingredient442 - ingredient
      cocktail486 - cocktail
      dispenser378 dispenser33 dispenser190 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker486)
  (ontable shot7)
  (ontable shot212)
  (dispenses dispenser378 ingredient489)
  (dispenses dispenser33 ingredient38)
  (dispenses dispenser190 ingredient442)
  (clean shaker486)
  (clean shot7)
  (clean shot212)
  (empty shaker486)
  (empty shot7)
  (empty shot212)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker486 l0)
  (shaker-level shaker486 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail486 ingredient38)
  (cocktail-part2 cocktail486 ingredient442)
)
 (:goal
  (and
      (contains shot7 cocktail486)
)))
