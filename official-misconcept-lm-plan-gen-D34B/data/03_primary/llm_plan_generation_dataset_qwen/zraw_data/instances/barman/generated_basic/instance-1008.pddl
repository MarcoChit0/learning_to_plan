(define (problem prob)
 (:domain barman)
 (:objects 
      shaker186 - shaker
      left right - hand
      shot269 shot352 - shot
      ingredient50 ingredient277 - ingredient
      cocktail1 - cocktail
      dispenser167 dispenser225 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker186)
  (ontable shot269)
  (ontable shot352)
  (dispenses dispenser167 ingredient50)
  (dispenses dispenser225 ingredient277)
  (clean shaker186)
  (clean shot269)
  (clean shot352)
  (empty shaker186)
  (empty shot269)
  (empty shot352)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker186 l0)
  (shaker-level shaker186 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient277)
  (cocktail-part2 cocktail1 ingredient50)
)
 (:goal
  (and
      (contains shot269 cocktail1)
)))
