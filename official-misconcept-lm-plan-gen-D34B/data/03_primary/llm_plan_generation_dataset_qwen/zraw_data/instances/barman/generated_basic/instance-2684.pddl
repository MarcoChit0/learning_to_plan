(define (problem prob)
 (:domain barman)
 (:objects 
      shaker141 - shaker
      left right - hand
      shot482 shot217 - shot
      ingredient117 ingredient31 ingredient358 - ingredient
      cocktail331 - cocktail
      dispenser122 dispenser390 dispenser342 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker141)
  (ontable shot482)
  (ontable shot217)
  (dispenses dispenser122 ingredient117)
  (dispenses dispenser390 ingredient31)
  (dispenses dispenser342 ingredient358)
  (clean shaker141)
  (clean shot482)
  (clean shot217)
  (empty shaker141)
  (empty shot482)
  (empty shot217)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker141 l0)
  (shaker-level shaker141 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail331 ingredient117)
  (cocktail-part2 cocktail331 ingredient31)
)
 (:goal
  (and
      (contains shot482 cocktail331)
)))
