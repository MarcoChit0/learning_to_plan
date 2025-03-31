(define (problem prob)
 (:domain barman)
 (:objects 
      shaker361 - shaker
      left right - hand
      shot327 - shot
      ingredient70 ingredient352 ingredient65 ingredient486 - ingredient
      cocktail1 - cocktail
      dispenser438 dispenser236 dispenser317 dispenser478 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker361)
  (ontable shot327)
  (dispenses dispenser438 ingredient70)
  (dispenses dispenser236 ingredient352)
  (dispenses dispenser317 ingredient65)
  (dispenses dispenser478 ingredient486)
  (clean shaker361)
  (clean shot327)
  (empty shaker361)
  (empty shot327)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker361 l0)
  (shaker-level shaker361 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient352)
  (cocktail-part2 cocktail1 ingredient65)
)
 (:goal
  (and
      (contains shot327 cocktail1)
)))
