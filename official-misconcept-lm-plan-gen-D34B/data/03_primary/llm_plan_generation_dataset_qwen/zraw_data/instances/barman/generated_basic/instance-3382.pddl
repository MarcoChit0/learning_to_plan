(define (problem prob)
 (:domain barman)
 (:objects 
      shaker310 - shaker
      left right - hand
      shot32 shot402 shot320 - shot
      ingredient376 ingredient206 ingredient338 - ingredient
      cocktail23 - cocktail
      dispenser399 dispenser171 dispenser24 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker310)
  (ontable shot32)
  (ontable shot402)
  (ontable shot320)
  (dispenses dispenser399 ingredient376)
  (dispenses dispenser171 ingredient206)
  (dispenses dispenser24 ingredient338)
  (clean shaker310)
  (clean shot32)
  (clean shot402)
  (clean shot320)
  (empty shaker310)
  (empty shot32)
  (empty shot402)
  (empty shot320)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker310 l0)
  (shaker-level shaker310 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail23 ingredient206)
  (cocktail-part2 cocktail23 ingredient376)
)
 (:goal
  (and
      (contains shot32 cocktail23)
      (contains shot402 cocktail23)
)))
