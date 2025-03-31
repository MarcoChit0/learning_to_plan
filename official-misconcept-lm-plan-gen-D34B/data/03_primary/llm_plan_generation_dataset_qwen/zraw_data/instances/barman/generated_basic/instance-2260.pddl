(define (problem prob)
 (:domain barman)
 (:objects 
      shaker481 - shaker
      left right - hand
      shot170 shot359 - shot
      ingredient138 ingredient468 ingredient334 ingredient58 - ingredient
      cocktail211 - cocktail
      dispenser322 dispenser397 dispenser173 dispenser304 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker481)
  (ontable shot170)
  (ontable shot359)
  (dispenses dispenser322 ingredient138)
  (dispenses dispenser397 ingredient468)
  (dispenses dispenser173 ingredient334)
  (dispenses dispenser304 ingredient58)
  (clean shaker481)
  (clean shot170)
  (clean shot359)
  (empty shaker481)
  (empty shot170)
  (empty shot359)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker481 l0)
  (shaker-level shaker481 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail211 ingredient334)
  (cocktail-part2 cocktail211 ingredient138)
)
 (:goal
  (and
      (contains shot170 cocktail211)
)))
