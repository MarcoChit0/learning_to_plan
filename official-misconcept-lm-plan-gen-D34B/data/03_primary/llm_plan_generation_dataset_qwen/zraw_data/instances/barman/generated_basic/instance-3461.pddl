(define (problem prob)
 (:domain barman)
 (:objects 
      shaker18 - shaker
      left right - hand
      shot339 shot312 shot400 - shot
      ingredient298 ingredient34 ingredient118 ingredient80 - ingredient
      cocktail412 - cocktail
      dispenser292 dispenser486 dispenser401 dispenser313 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker18)
  (ontable shot339)
  (ontable shot312)
  (ontable shot400)
  (dispenses dispenser292 ingredient298)
  (dispenses dispenser486 ingredient34)
  (dispenses dispenser401 ingredient118)
  (dispenses dispenser313 ingredient80)
  (clean shaker18)
  (clean shot339)
  (clean shot312)
  (clean shot400)
  (empty shaker18)
  (empty shot339)
  (empty shot312)
  (empty shot400)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker18 l0)
  (shaker-level shaker18 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail412 ingredient80)
  (cocktail-part2 cocktail412 ingredient118)
)
 (:goal
  (and
      (contains shot339 cocktail412)
      (contains shot312 ingredient80)
)))
