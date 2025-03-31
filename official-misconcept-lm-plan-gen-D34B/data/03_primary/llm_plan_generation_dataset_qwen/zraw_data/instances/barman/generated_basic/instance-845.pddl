(define (problem prob)
 (:domain barman)
 (:objects 
      shaker210 - shaker
      left right - hand
      shot248 shot494 shot259 - shot
      ingredient299 ingredient309 ingredient225 ingredient204 - ingredient
      cocktail1 - cocktail
      dispenser393 dispenser27 dispenser407 dispenser208 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker210)
  (ontable shot248)
  (ontable shot494)
  (ontable shot259)
  (dispenses dispenser393 ingredient299)
  (dispenses dispenser27 ingredient309)
  (dispenses dispenser407 ingredient225)
  (dispenses dispenser208 ingredient204)
  (clean shaker210)
  (clean shot248)
  (clean shot494)
  (clean shot259)
  (empty shaker210)
  (empty shot248)
  (empty shot494)
  (empty shot259)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker210 l0)
  (shaker-level shaker210 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient299)
  (cocktail-part2 cocktail1 ingredient204)
)
 (:goal
  (and
      (contains shot248 cocktail1)
      (contains shot494 cocktail1)
)))
