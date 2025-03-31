(define (problem prob)
 (:domain barman)
 (:objects 
      shaker225 - shaker
      left right - hand
      shot441 - shot
      ingredient382 ingredient354 - ingredient
      cocktail1 - cocktail
      dispenser381 dispenser297 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker225)
  (ontable shot441)
  (dispenses dispenser381 ingredient382)
  (dispenses dispenser297 ingredient354)
  (clean shaker225)
  (clean shot441)
  (empty shaker225)
  (empty shot441)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker225 l0)
  (shaker-level shaker225 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient354)
  (cocktail-part2 cocktail1 ingredient382)
)
 (:goal
  (and
      (contains shot441 cocktail1)
)))
