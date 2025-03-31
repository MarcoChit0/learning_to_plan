(define (problem prob)
 (:domain barman)
 (:objects 
      shaker350 - shaker
      left right - hand
      shot275 shot74 - shot
      ingredient184 ingredient233 - ingredient
      cocktail1 - cocktail
      dispenser486 dispenser120 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker350)
  (ontable shot275)
  (ontable shot74)
  (dispenses dispenser486 ingredient184)
  (dispenses dispenser120 ingredient233)
  (clean shaker350)
  (clean shot275)
  (clean shot74)
  (empty shaker350)
  (empty shot275)
  (empty shot74)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker350 l0)
  (shaker-level shaker350 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient184)
  (cocktail-part2 cocktail1 ingredient233)
)
 (:goal
  (and
      (contains shot275 cocktail1)
)))
