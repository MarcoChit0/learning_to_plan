(define (problem prob)
 (:domain barman)
 (:objects 
      shaker482 - shaker
      left right - hand
      shot378 - shot
      ingredient96 ingredient46 ingredient300 - ingredient
      cocktail132 - cocktail
      dispenser103 dispenser42 dispenser483 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker482)
  (ontable shot378)
  (dispenses dispenser103 ingredient96)
  (dispenses dispenser42 ingredient46)
  (dispenses dispenser483 ingredient300)
  (clean shaker482)
  (clean shot378)
  (empty shaker482)
  (empty shot378)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker482 l0)
  (shaker-level shaker482 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail132 ingredient96)
  (cocktail-part2 cocktail132 ingredient300)
)
 (:goal
  (and
      (contains shot378 cocktail132)
)))
