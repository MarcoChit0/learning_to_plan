(define (problem prob)
 (:domain barman)
 (:objects 
      shaker36 - shaker
      left right - hand
      shot359 - shot
      ingredient259 ingredient120 - ingredient
      cocktail1 - cocktail
      dispenser53 dispenser154 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker36)
  (ontable shot359)
  (dispenses dispenser53 ingredient259)
  (dispenses dispenser154 ingredient120)
  (clean shaker36)
  (clean shot359)
  (empty shaker36)
  (empty shot359)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker36 l0)
  (shaker-level shaker36 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient259)
  (cocktail-part2 cocktail1 ingredient120)
)
 (:goal
  (and
      (contains shot359 cocktail1)
)))
