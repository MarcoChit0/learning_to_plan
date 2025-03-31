(define (problem prob)
 (:domain barman)
 (:objects 
      shaker478 - shaker
      left right - hand
      shot93 shot117 shot238 - shot
      ingredient292 ingredient93 ingredient480 ingredient493 - ingredient
      cocktail1 - cocktail
      dispenser34 dispenser176 dispenser421 dispenser296 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker478)
  (ontable shot93)
  (ontable shot117)
  (ontable shot238)
  (dispenses dispenser34 ingredient292)
  (dispenses dispenser176 ingredient93)
  (dispenses dispenser421 ingredient480)
  (dispenses dispenser296 ingredient493)
  (clean shaker478)
  (clean shot93)
  (clean shot117)
  (clean shot238)
  (empty shaker478)
  (empty shot93)
  (empty shot117)
  (empty shot238)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker478 l0)
  (shaker-level shaker478 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient493)
  (cocktail-part2 cocktail1 ingredient292)
)
 (:goal
  (and
      (contains shot93 cocktail1)
      (contains shot117 cocktail1)
)))
