(define (problem prob)
 (:domain barman)
 (:objects 
      shaker316 - shaker
      left right - hand
      shot453 - shot
      ingredient323 ingredient14 - ingredient
      cocktail1 - cocktail
      dispenser152 dispenser369 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker316)
  (ontable shot453)
  (dispenses dispenser152 ingredient323)
  (dispenses dispenser369 ingredient14)
  (clean shaker316)
  (clean shot453)
  (empty shaker316)
  (empty shot453)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker316 l0)
  (shaker-level shaker316 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient14)
  (cocktail-part2 cocktail1 ingredient323)
)
 (:goal
  (and
      (contains shot453 cocktail1)
)))
