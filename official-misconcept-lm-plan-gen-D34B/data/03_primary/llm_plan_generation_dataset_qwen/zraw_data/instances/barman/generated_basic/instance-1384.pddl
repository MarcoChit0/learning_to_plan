(define (problem prob)
 (:domain barman)
 (:objects 
      shaker217 - shaker
      left right - hand
      shot118 shot320 shot12 - shot
      ingredient246 ingredient205 ingredient243 ingredient43 - ingredient
      cocktail371 - cocktail
      dispenser63 dispenser477 dispenser53 dispenser107 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker217)
  (ontable shot118)
  (ontable shot320)
  (ontable shot12)
  (dispenses dispenser63 ingredient246)
  (dispenses dispenser477 ingredient205)
  (dispenses dispenser53 ingredient243)
  (dispenses dispenser107 ingredient43)
  (clean shaker217)
  (clean shot118)
  (clean shot320)
  (clean shot12)
  (empty shaker217)
  (empty shot118)
  (empty shot320)
  (empty shot12)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker217 l0)
  (shaker-level shaker217 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail371 ingredient243)
  (cocktail-part2 cocktail371 ingredient205)
)
 (:goal
  (and
      (contains shot118 cocktail371)
      (contains shot320 ingredient43)
)))
