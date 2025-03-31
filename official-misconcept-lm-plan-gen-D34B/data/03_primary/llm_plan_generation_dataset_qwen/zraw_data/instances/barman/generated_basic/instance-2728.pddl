(define (problem prob)
 (:domain barman)
 (:objects 
      shaker179 - shaker
      left right - hand
      shot46 shot12 shot230 - shot
      ingredient335 ingredient490 ingredient135 - ingredient
      cocktail390 - cocktail
      dispenser435 dispenser376 dispenser143 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker179)
  (ontable shot46)
  (ontable shot12)
  (ontable shot230)
  (dispenses dispenser435 ingredient335)
  (dispenses dispenser376 ingredient490)
  (dispenses dispenser143 ingredient135)
  (clean shaker179)
  (clean shot46)
  (clean shot12)
  (clean shot230)
  (empty shaker179)
  (empty shot46)
  (empty shot12)
  (empty shot230)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker179 l0)
  (shaker-level shaker179 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail390 ingredient335)
  (cocktail-part2 cocktail390 ingredient135)
)
 (:goal
  (and
      (contains shot46 cocktail390)
      (contains shot12 ingredient490)
)))
