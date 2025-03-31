(define (problem prob)
 (:domain barman)
 (:objects 
      shaker407 - shaker
      left right - hand
      shot12 shot70 - shot
      ingredient426 ingredient258 ingredient320 - ingredient
      cocktail251 - cocktail
      dispenser91 dispenser112 dispenser48 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker407)
  (ontable shot12)
  (ontable shot70)
  (dispenses dispenser91 ingredient426)
  (dispenses dispenser112 ingredient258)
  (dispenses dispenser48 ingredient320)
  (clean shaker407)
  (clean shot12)
  (clean shot70)
  (empty shaker407)
  (empty shot12)
  (empty shot70)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker407 l0)
  (shaker-level shaker407 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail251 ingredient426)
  (cocktail-part2 cocktail251 ingredient258)
)
 (:goal
  (and
      (contains shot12 cocktail251)
)))
