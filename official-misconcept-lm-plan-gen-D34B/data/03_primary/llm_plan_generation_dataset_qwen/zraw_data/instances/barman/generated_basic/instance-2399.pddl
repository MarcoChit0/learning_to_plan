(define (problem prob)
 (:domain barman)
 (:objects 
      shaker248 - shaker
      left right - hand
      shot448 shot73 - shot
      ingredient444 ingredient382 ingredient60 - ingredient
      cocktail214 - cocktail
      dispenser106 dispenser479 dispenser471 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker248)
  (ontable shot448)
  (ontable shot73)
  (dispenses dispenser106 ingredient444)
  (dispenses dispenser479 ingredient382)
  (dispenses dispenser471 ingredient60)
  (clean shaker248)
  (clean shot448)
  (clean shot73)
  (empty shaker248)
  (empty shot448)
  (empty shot73)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker248 l0)
  (shaker-level shaker248 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail214 ingredient60)
  (cocktail-part2 cocktail214 ingredient444)
)
 (:goal
  (and
      (contains shot448 cocktail214)
)))
