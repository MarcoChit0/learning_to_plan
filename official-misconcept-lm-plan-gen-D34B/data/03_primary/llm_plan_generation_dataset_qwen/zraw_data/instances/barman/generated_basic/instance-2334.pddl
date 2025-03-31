(define (problem prob)
 (:domain barman)
 (:objects 
      shaker41 - shaker
      left right - hand
      shot211 - shot
      ingredient424 ingredient499 - ingredient
      cocktail381 - cocktail
      dispenser391 dispenser242 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker41)
  (ontable shot211)
  (dispenses dispenser391 ingredient424)
  (dispenses dispenser242 ingredient499)
  (clean shaker41)
  (clean shot211)
  (empty shaker41)
  (empty shot211)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker41 l0)
  (shaker-level shaker41 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail381 ingredient499)
  (cocktail-part2 cocktail381 ingredient424)
)
 (:goal
  (and
      (contains shot211 cocktail381)
)))
