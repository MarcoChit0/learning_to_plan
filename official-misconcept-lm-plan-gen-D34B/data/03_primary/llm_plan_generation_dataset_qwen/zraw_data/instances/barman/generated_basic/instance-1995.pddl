(define (problem prob)
 (:domain barman)
 (:objects 
      shaker186 - shaker
      left right - hand
      shot39 shot245 - shot
      ingredient466 ingredient45 ingredient407 - ingredient
      cocktail69 - cocktail
      dispenser140 dispenser488 dispenser128 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker186)
  (ontable shot39)
  (ontable shot245)
  (dispenses dispenser140 ingredient466)
  (dispenses dispenser488 ingredient45)
  (dispenses dispenser128 ingredient407)
  (clean shaker186)
  (clean shot39)
  (clean shot245)
  (empty shaker186)
  (empty shot39)
  (empty shot245)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker186 l0)
  (shaker-level shaker186 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail69 ingredient407)
  (cocktail-part2 cocktail69 ingredient466)
)
 (:goal
  (and
      (contains shot39 cocktail69)
)))
