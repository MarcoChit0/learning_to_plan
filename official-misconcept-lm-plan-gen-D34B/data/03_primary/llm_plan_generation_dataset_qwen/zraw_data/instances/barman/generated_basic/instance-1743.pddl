(define (problem prob)
 (:domain barman)
 (:objects 
      shaker313 - shaker
      left right - hand
      shot178 shot217 - shot
      ingredient40 ingredient451 ingredient406 - ingredient
      cocktail211 - cocktail
      dispenser408 dispenser87 dispenser187 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker313)
  (ontable shot178)
  (ontable shot217)
  (dispenses dispenser408 ingredient40)
  (dispenses dispenser87 ingredient451)
  (dispenses dispenser187 ingredient406)
  (clean shaker313)
  (clean shot178)
  (clean shot217)
  (empty shaker313)
  (empty shot178)
  (empty shot217)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker313 l0)
  (shaker-level shaker313 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail211 ingredient406)
  (cocktail-part2 cocktail211 ingredient40)
)
 (:goal
  (and
      (contains shot178 cocktail211)
)))
