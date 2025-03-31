(define (problem prob)
 (:domain barman)
 (:objects 
      shaker167 - shaker
      left right - hand
      shot199 shot113 shot384 - shot
      ingredient176 ingredient473 ingredient59 ingredient313 - ingredient
      cocktail32 - cocktail
      dispenser300 dispenser482 dispenser122 dispenser109 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker167)
  (ontable shot199)
  (ontable shot113)
  (ontable shot384)
  (dispenses dispenser300 ingredient176)
  (dispenses dispenser482 ingredient473)
  (dispenses dispenser122 ingredient59)
  (dispenses dispenser109 ingredient313)
  (clean shaker167)
  (clean shot199)
  (clean shot113)
  (clean shot384)
  (empty shaker167)
  (empty shot199)
  (empty shot113)
  (empty shot384)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker167 l0)
  (shaker-level shaker167 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail32 ingredient176)
  (cocktail-part2 cocktail32 ingredient59)
)
 (:goal
  (and
      (contains shot199 cocktail32)
      (contains shot113 ingredient313)
)))
