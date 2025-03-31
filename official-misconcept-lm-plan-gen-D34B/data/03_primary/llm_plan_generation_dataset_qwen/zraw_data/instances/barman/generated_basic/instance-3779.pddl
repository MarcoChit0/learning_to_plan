(define (problem prob)
 (:domain barman)
 (:objects 
      shaker469 - shaker
      left right - hand
      shot179 shot211 shot357 - shot
      ingredient471 ingredient340 - ingredient
      cocktail319 - cocktail
      dispenser400 dispenser100 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker469)
  (ontable shot179)
  (ontable shot211)
  (ontable shot357)
  (dispenses dispenser400 ingredient471)
  (dispenses dispenser100 ingredient340)
  (clean shaker469)
  (clean shot179)
  (clean shot211)
  (clean shot357)
  (empty shaker469)
  (empty shot179)
  (empty shot211)
  (empty shot357)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker469 l0)
  (shaker-level shaker469 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail319 ingredient340)
  (cocktail-part2 cocktail319 ingredient471)
)
 (:goal
  (and
      (contains shot179 cocktail319)
      (contains shot211 cocktail319)
)))
