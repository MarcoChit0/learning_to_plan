(define (problem prob)
 (:domain barman)
 (:objects 
      shaker459 - shaker
      left right - hand
      shot230 shot32 - shot
      ingredient332 ingredient206 ingredient167 - ingredient
      cocktail181 - cocktail
      dispenser49 dispenser333 dispenser343 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker459)
  (ontable shot230)
  (ontable shot32)
  (dispenses dispenser49 ingredient332)
  (dispenses dispenser333 ingredient206)
  (dispenses dispenser343 ingredient167)
  (clean shaker459)
  (clean shot230)
  (clean shot32)
  (empty shaker459)
  (empty shot230)
  (empty shot32)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker459 l0)
  (shaker-level shaker459 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail181 ingredient332)
  (cocktail-part2 cocktail181 ingredient206)
)
 (:goal
  (and
      (contains shot230 cocktail181)
)))
