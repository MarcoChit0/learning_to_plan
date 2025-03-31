(define (problem prob)
 (:domain barman)
 (:objects 
      shaker194 - shaker
      left right - hand
      shot190 shot194 shot56 - shot
      ingredient106 ingredient193 ingredient343 - ingredient
      cocktail350 - cocktail
      dispenser151 dispenser206 dispenser410 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker194)
  (ontable shot190)
  (ontable shot194)
  (ontable shot56)
  (dispenses dispenser151 ingredient106)
  (dispenses dispenser206 ingredient193)
  (dispenses dispenser410 ingredient343)
  (clean shaker194)
  (clean shot190)
  (clean shot194)
  (clean shot56)
  (empty shaker194)
  (empty shot190)
  (empty shot194)
  (empty shot56)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker194 l0)
  (shaker-level shaker194 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail350 ingredient343)
  (cocktail-part2 cocktail350 ingredient106)
)
 (:goal
  (and
      (contains shot190 cocktail350)
      (contains shot194 cocktail350)
)))
