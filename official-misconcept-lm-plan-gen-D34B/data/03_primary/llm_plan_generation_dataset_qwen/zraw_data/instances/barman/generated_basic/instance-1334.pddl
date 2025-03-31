(define (problem prob)
 (:domain barman)
 (:objects 
      shaker465 - shaker
      left right - hand
      shot117 shot39 shot367 - shot
      ingredient401 ingredient115 ingredient436 - ingredient
      cocktail407 - cocktail
      dispenser301 dispenser398 dispenser487 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker465)
  (ontable shot117)
  (ontable shot39)
  (ontable shot367)
  (dispenses dispenser301 ingredient401)
  (dispenses dispenser398 ingredient115)
  (dispenses dispenser487 ingredient436)
  (clean shaker465)
  (clean shot117)
  (clean shot39)
  (clean shot367)
  (empty shaker465)
  (empty shot117)
  (empty shot39)
  (empty shot367)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker465 l0)
  (shaker-level shaker465 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail407 ingredient115)
  (cocktail-part2 cocktail407 ingredient401)
)
 (:goal
  (and
      (contains shot117 cocktail407)
      (contains shot39 ingredient401)
)))
