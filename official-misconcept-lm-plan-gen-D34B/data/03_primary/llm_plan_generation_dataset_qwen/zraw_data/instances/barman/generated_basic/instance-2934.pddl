(define (problem prob)
 (:domain barman)
 (:objects 
      shaker295 - shaker
      left right - hand
      shot387 - shot
      ingredient338 ingredient290 - ingredient
      cocktail228 - cocktail
      dispenser347 dispenser383 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker295)
  (ontable shot387)
  (dispenses dispenser347 ingredient338)
  (dispenses dispenser383 ingredient290)
  (clean shaker295)
  (clean shot387)
  (empty shaker295)
  (empty shot387)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker295 l0)
  (shaker-level shaker295 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail228 ingredient338)
  (cocktail-part2 cocktail228 ingredient290)
)
 (:goal
  (and
      (contains shot387 cocktail228)
)))
