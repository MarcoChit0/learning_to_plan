(define (problem prob)
 (:domain barman)
 (:objects 
      shaker424 - shaker
      left right - hand
      shot41 shot370 shot46 - shot
      ingredient85 ingredient309 ingredient131 ingredient184 - ingredient
      cocktail140 - cocktail
      dispenser162 dispenser253 dispenser100 dispenser453 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker424)
  (ontable shot41)
  (ontable shot370)
  (ontable shot46)
  (dispenses dispenser162 ingredient85)
  (dispenses dispenser253 ingredient309)
  (dispenses dispenser100 ingredient131)
  (dispenses dispenser453 ingredient184)
  (clean shaker424)
  (clean shot41)
  (clean shot370)
  (clean shot46)
  (empty shaker424)
  (empty shot41)
  (empty shot370)
  (empty shot46)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker424 l0)
  (shaker-level shaker424 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail140 ingredient85)
  (cocktail-part2 cocktail140 ingredient184)
)
 (:goal
  (and
      (contains shot41 cocktail140)
      (contains shot370 cocktail140)
)))
