(define (problem prob)
 (:domain barman)
 (:objects 
      shaker441 - shaker
      left right - hand
      shot406 shot7 - shot
      ingredient237 ingredient284 ingredient69 - ingredient
      cocktail421 - cocktail
      dispenser338 dispenser185 dispenser32 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker441)
  (ontable shot406)
  (ontable shot7)
  (dispenses dispenser338 ingredient237)
  (dispenses dispenser185 ingredient284)
  (dispenses dispenser32 ingredient69)
  (clean shaker441)
  (clean shot406)
  (clean shot7)
  (empty shaker441)
  (empty shot406)
  (empty shot7)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker441 l0)
  (shaker-level shaker441 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail421 ingredient284)
  (cocktail-part2 cocktail421 ingredient237)
)
 (:goal
  (and
      (contains shot406 cocktail421)
)))
