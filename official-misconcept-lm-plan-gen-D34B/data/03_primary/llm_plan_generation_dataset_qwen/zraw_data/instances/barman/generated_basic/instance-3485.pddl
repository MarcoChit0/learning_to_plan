(define (problem prob)
 (:domain barman)
 (:objects 
      shaker496 - shaker
      left right - hand
      shot479 shot252 shot53 - shot
      ingredient191 ingredient441 ingredient427 - ingredient
      cocktail288 - cocktail
      dispenser140 dispenser150 dispenser28 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker496)
  (ontable shot479)
  (ontable shot252)
  (ontable shot53)
  (dispenses dispenser140 ingredient191)
  (dispenses dispenser150 ingredient441)
  (dispenses dispenser28 ingredient427)
  (clean shaker496)
  (clean shot479)
  (clean shot252)
  (clean shot53)
  (empty shaker496)
  (empty shot479)
  (empty shot252)
  (empty shot53)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker496 l0)
  (shaker-level shaker496 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail288 ingredient441)
  (cocktail-part2 cocktail288 ingredient427)
)
 (:goal
  (and
      (contains shot479 cocktail288)
      (contains shot252 cocktail288)
)))
