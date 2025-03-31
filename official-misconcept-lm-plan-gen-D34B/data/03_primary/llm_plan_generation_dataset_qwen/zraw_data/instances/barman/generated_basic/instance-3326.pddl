(define (problem prob)
 (:domain barman)
 (:objects 
      shaker238 - shaker
      left right - hand
      shot211 shot407 shot405 - shot
      ingredient234 ingredient429 ingredient117 - ingredient
      cocktail32 - cocktail
      dispenser212 dispenser424 dispenser359 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker238)
  (ontable shot211)
  (ontable shot407)
  (ontable shot405)
  (dispenses dispenser212 ingredient234)
  (dispenses dispenser424 ingredient429)
  (dispenses dispenser359 ingredient117)
  (clean shaker238)
  (clean shot211)
  (clean shot407)
  (clean shot405)
  (empty shaker238)
  (empty shot211)
  (empty shot407)
  (empty shot405)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker238 l0)
  (shaker-level shaker238 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail32 ingredient429)
  (cocktail-part2 cocktail32 ingredient117)
)
 (:goal
  (and
      (contains shot211 cocktail32)
      (contains shot407 cocktail32)
)))
