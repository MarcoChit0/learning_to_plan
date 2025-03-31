(define (problem prob)
 (:domain barman)
 (:objects 
      shaker475 - shaker
      left right - hand
      shot398 - shot
      ingredient499 ingredient237 - ingredient
      cocktail194 - cocktail
      dispenser341 dispenser402 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker475)
  (ontable shot398)
  (dispenses dispenser341 ingredient499)
  (dispenses dispenser402 ingredient237)
  (clean shaker475)
  (clean shot398)
  (empty shaker475)
  (empty shot398)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker475 l0)
  (shaker-level shaker475 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail194 ingredient499)
  (cocktail-part2 cocktail194 ingredient237)
)
 (:goal
  (and
      (contains shot398 cocktail194)
)))
