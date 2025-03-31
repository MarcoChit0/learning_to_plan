(define (problem prob)
 (:domain barman)
 (:objects 
      shaker59 - shaker
      left right - hand
      shot118 - shot
      ingredient373 ingredient364 - ingredient
      cocktail257 - cocktail
      dispenser489 dispenser42 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker59)
  (ontable shot118)
  (dispenses dispenser489 ingredient373)
  (dispenses dispenser42 ingredient364)
  (clean shaker59)
  (clean shot118)
  (empty shaker59)
  (empty shot118)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker59 l0)
  (shaker-level shaker59 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail257 ingredient373)
  (cocktail-part2 cocktail257 ingredient364)
)
 (:goal
  (and
      (contains shot118 cocktail257)
)))
