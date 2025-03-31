(define (problem prob)
 (:domain barman)
 (:objects 
      shaker427 - shaker
      left right - hand
      shot185 shot460 - shot
      ingredient65 ingredient184 ingredient212 - ingredient
      cocktail1 - cocktail
      dispenser50 dispenser490 dispenser306 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker427)
  (ontable shot185)
  (ontable shot460)
  (dispenses dispenser50 ingredient65)
  (dispenses dispenser490 ingredient184)
  (dispenses dispenser306 ingredient212)
  (clean shaker427)
  (clean shot185)
  (clean shot460)
  (empty shaker427)
  (empty shot185)
  (empty shot460)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker427 l0)
  (shaker-level shaker427 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient212)
  (cocktail-part2 cocktail1 ingredient65)
)
 (:goal
  (and
      (contains shot185 cocktail1)
)))
