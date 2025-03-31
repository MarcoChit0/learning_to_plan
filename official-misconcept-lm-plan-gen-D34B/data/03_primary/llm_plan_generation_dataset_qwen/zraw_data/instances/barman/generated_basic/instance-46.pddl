(define (problem prob)
 (:domain barman)
 (:objects 
      shaker123 - shaker
      left right - hand
      shot65 shot432 - shot
      ingredient139 ingredient389 ingredient21 ingredient249 - ingredient
      cocktail1 - cocktail
      dispenser312 dispenser416 dispenser314 dispenser233 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker123)
  (ontable shot65)
  (ontable shot432)
  (dispenses dispenser312 ingredient139)
  (dispenses dispenser416 ingredient389)
  (dispenses dispenser314 ingredient21)
  (dispenses dispenser233 ingredient249)
  (clean shaker123)
  (clean shot65)
  (clean shot432)
  (empty shaker123)
  (empty shot65)
  (empty shot432)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker123 l0)
  (shaker-level shaker123 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient389)
  (cocktail-part2 cocktail1 ingredient249)
)
 (:goal
  (and
      (contains shot65 cocktail1)
)))
