(define (problem prob)
 (:domain barman)
 (:objects 
      shaker372 - shaker
      left right - hand
      shot54 shot426 - shot
      ingredient184 ingredient187 ingredient94 - ingredient
      cocktail1 - cocktail
      dispenser312 dispenser276 dispenser347 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker372)
  (ontable shot54)
  (ontable shot426)
  (dispenses dispenser312 ingredient184)
  (dispenses dispenser276 ingredient187)
  (dispenses dispenser347 ingredient94)
  (clean shaker372)
  (clean shot54)
  (clean shot426)
  (empty shaker372)
  (empty shot54)
  (empty shot426)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker372 l0)
  (shaker-level shaker372 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient184)
  (cocktail-part2 cocktail1 ingredient94)
)
 (:goal
  (and
      (contains shot54 cocktail1)
)))
