(define (problem prob)
 (:domain barman)
 (:objects 
      shaker120 - shaker
      left right - hand
      shot70 shot250 - shot
      ingredient413 ingredient69 ingredient227 ingredient187 - ingredient
      cocktail1 - cocktail
      dispenser232 dispenser245 dispenser21 dispenser387 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker120)
  (ontable shot70)
  (ontable shot250)
  (dispenses dispenser232 ingredient413)
  (dispenses dispenser245 ingredient69)
  (dispenses dispenser21 ingredient227)
  (dispenses dispenser387 ingredient187)
  (clean shaker120)
  (clean shot70)
  (clean shot250)
  (empty shaker120)
  (empty shot70)
  (empty shot250)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker120 l0)
  (shaker-level shaker120 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient227)
  (cocktail-part2 cocktail1 ingredient69)
)
 (:goal
  (and
      (contains shot70 cocktail1)
)))
