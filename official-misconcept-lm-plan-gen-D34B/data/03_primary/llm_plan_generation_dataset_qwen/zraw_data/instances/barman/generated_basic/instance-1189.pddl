(define (problem prob)
 (:domain barman)
 (:objects 
      shaker62 - shaker
      left right - hand
      shot266 shot344 - shot
      ingredient222 ingredient204 - ingredient
      cocktail83 - cocktail
      dispenser97 dispenser292 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker62)
  (ontable shot266)
  (ontable shot344)
  (dispenses dispenser97 ingredient222)
  (dispenses dispenser292 ingredient204)
  (clean shaker62)
  (clean shot266)
  (clean shot344)
  (empty shaker62)
  (empty shot266)
  (empty shot344)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker62 l0)
  (shaker-level shaker62 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail83 ingredient222)
  (cocktail-part2 cocktail83 ingredient204)
)
 (:goal
  (and
      (contains shot266 cocktail83)
)))
