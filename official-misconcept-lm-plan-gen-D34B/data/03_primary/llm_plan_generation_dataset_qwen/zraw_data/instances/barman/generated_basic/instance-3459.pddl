(define (problem prob)
 (:domain barman)
 (:objects 
      shaker8 - shaker
      left right - hand
      shot434 shot32 shot346 - shot
      ingredient316 ingredient50 ingredient357 ingredient451 - ingredient
      cocktail250 - cocktail
      dispenser452 dispenser477 dispenser24 dispenser480 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker8)
  (ontable shot434)
  (ontable shot32)
  (ontable shot346)
  (dispenses dispenser452 ingredient316)
  (dispenses dispenser477 ingredient50)
  (dispenses dispenser24 ingredient357)
  (dispenses dispenser480 ingredient451)
  (clean shaker8)
  (clean shot434)
  (clean shot32)
  (clean shot346)
  (empty shaker8)
  (empty shot434)
  (empty shot32)
  (empty shot346)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker8 l0)
  (shaker-level shaker8 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail250 ingredient451)
  (cocktail-part2 cocktail250 ingredient316)
)
 (:goal
  (and
      (contains shot434 cocktail250)
      (contains shot32 cocktail250)
)))
