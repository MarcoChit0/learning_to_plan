(define (problem prob)
 (:domain barman)
 (:objects 
      shaker481 - shaker
      left right - hand
      shot249 shot194 - shot
      ingredient224 ingredient484 ingredient396 ingredient127 - ingredient
      cocktail109 - cocktail
      dispenser402 dispenser159 dispenser224 dispenser87 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker481)
  (ontable shot249)
  (ontable shot194)
  (dispenses dispenser402 ingredient224)
  (dispenses dispenser159 ingredient484)
  (dispenses dispenser224 ingredient396)
  (dispenses dispenser87 ingredient127)
  (clean shaker481)
  (clean shot249)
  (clean shot194)
  (empty shaker481)
  (empty shot249)
  (empty shot194)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker481 l0)
  (shaker-level shaker481 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail109 ingredient224)
  (cocktail-part2 cocktail109 ingredient396)
)
 (:goal
  (and
      (contains shot249 cocktail109)
)))
