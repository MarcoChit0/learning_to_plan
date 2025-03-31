(define (problem prob)
 (:domain barman)
 (:objects 
      shaker439 - shaker
      left right - hand
      shot96 - shot
      ingredient483 ingredient415 ingredient214 - ingredient
      cocktail1 - cocktail
      dispenser97 dispenser214 dispenser293 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker439)
  (ontable shot96)
  (dispenses dispenser97 ingredient483)
  (dispenses dispenser214 ingredient415)
  (dispenses dispenser293 ingredient214)
  (clean shaker439)
  (clean shot96)
  (empty shaker439)
  (empty shot96)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker439 l0)
  (shaker-level shaker439 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient214)
  (cocktail-part2 cocktail1 ingredient415)
)
 (:goal
  (and
      (contains shot96 cocktail1)
)))
