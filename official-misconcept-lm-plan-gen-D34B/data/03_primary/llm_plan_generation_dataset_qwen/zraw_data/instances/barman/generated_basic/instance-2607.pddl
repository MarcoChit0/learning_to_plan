(define (problem prob)
 (:domain barman)
 (:objects 
      shaker156 - shaker
      left right - hand
      shot163 shot135 - shot
      ingredient130 ingredient180 ingredient109 - ingredient
      cocktail162 - cocktail
      dispenser51 dispenser387 dispenser453 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker156)
  (ontable shot163)
  (ontable shot135)
  (dispenses dispenser51 ingredient130)
  (dispenses dispenser387 ingredient180)
  (dispenses dispenser453 ingredient109)
  (clean shaker156)
  (clean shot163)
  (clean shot135)
  (empty shaker156)
  (empty shot163)
  (empty shot135)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker156 l0)
  (shaker-level shaker156 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail162 ingredient180)
  (cocktail-part2 cocktail162 ingredient109)
)
 (:goal
  (and
      (contains shot163 cocktail162)
)))
