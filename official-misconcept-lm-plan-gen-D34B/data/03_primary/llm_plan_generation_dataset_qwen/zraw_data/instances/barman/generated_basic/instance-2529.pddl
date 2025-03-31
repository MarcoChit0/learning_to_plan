(define (problem prob)
 (:domain barman)
 (:objects 
      shaker113 - shaker
      left right - hand
      shot344 shot32 shot66 - shot
      ingredient315 ingredient406 - ingredient
      cocktail113 - cocktail
      dispenser31 dispenser409 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker113)
  (ontable shot344)
  (ontable shot32)
  (ontable shot66)
  (dispenses dispenser31 ingredient315)
  (dispenses dispenser409 ingredient406)
  (clean shaker113)
  (clean shot344)
  (clean shot32)
  (clean shot66)
  (empty shaker113)
  (empty shot344)
  (empty shot32)
  (empty shot66)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker113 l0)
  (shaker-level shaker113 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail113 ingredient406)
  (cocktail-part2 cocktail113 ingredient315)
)
 (:goal
  (and
      (contains shot344 cocktail113)
      (contains shot32 cocktail113)
)))
