(define (problem prob)
 (:domain barman)
 (:objects 
      shaker172 - shaker
      left right - hand
      shot116 shot401 - shot
      ingredient155 ingredient355 ingredient205 ingredient353 - ingredient
      cocktail290 - cocktail
      dispenser295 dispenser236 dispenser220 dispenser65 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker172)
  (ontable shot116)
  (ontable shot401)
  (dispenses dispenser295 ingredient155)
  (dispenses dispenser236 ingredient355)
  (dispenses dispenser220 ingredient205)
  (dispenses dispenser65 ingredient353)
  (clean shaker172)
  (clean shot116)
  (clean shot401)
  (empty shaker172)
  (empty shot116)
  (empty shot401)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker172 l0)
  (shaker-level shaker172 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail290 ingredient353)
  (cocktail-part2 cocktail290 ingredient205)
)
 (:goal
  (and
      (contains shot116 cocktail290)
)))
