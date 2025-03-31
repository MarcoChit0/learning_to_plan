(define (problem prob)
 (:domain barman)
 (:objects 
      shaker320 - shaker
      left right - hand
      shot133 shot42 shot281 - shot
      ingredient313 ingredient408 - ingredient
      cocktail292 - cocktail
      dispenser93 dispenser477 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker320)
  (ontable shot133)
  (ontable shot42)
  (ontable shot281)
  (dispenses dispenser93 ingredient313)
  (dispenses dispenser477 ingredient408)
  (clean shaker320)
  (clean shot133)
  (clean shot42)
  (clean shot281)
  (empty shaker320)
  (empty shot133)
  (empty shot42)
  (empty shot281)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker320 l0)
  (shaker-level shaker320 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail292 ingredient313)
  (cocktail-part2 cocktail292 ingredient408)
)
 (:goal
  (and
      (contains shot133 cocktail292)
      (contains shot42 cocktail292)
)))
