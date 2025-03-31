(define (problem prob)
 (:domain barman)
 (:objects 
      shaker103 - shaker
      left right - hand
      shot423 shot317 - shot
      ingredient206 ingredient280 ingredient487 - ingredient
      cocktail47 - cocktail
      dispenser477 dispenser103 dispenser12 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker103)
  (ontable shot423)
  (ontable shot317)
  (dispenses dispenser477 ingredient206)
  (dispenses dispenser103 ingredient280)
  (dispenses dispenser12 ingredient487)
  (clean shaker103)
  (clean shot423)
  (clean shot317)
  (empty shaker103)
  (empty shot423)
  (empty shot317)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker103 l0)
  (shaker-level shaker103 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail47 ingredient280)
  (cocktail-part2 cocktail47 ingredient206)
)
 (:goal
  (and
      (contains shot423 cocktail47)
)))
