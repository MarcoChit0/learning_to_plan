(define (problem prob)
 (:domain barman)
 (:objects 
      shaker33 - shaker
      left right - hand
      shot46 shot261 - shot
      ingredient146 ingredient475 ingredient234 ingredient36 - ingredient
      cocktail323 - cocktail
      dispenser219 dispenser365 dispenser348 dispenser16 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker33)
  (ontable shot46)
  (ontable shot261)
  (dispenses dispenser219 ingredient146)
  (dispenses dispenser365 ingredient475)
  (dispenses dispenser348 ingredient234)
  (dispenses dispenser16 ingredient36)
  (clean shaker33)
  (clean shot46)
  (clean shot261)
  (empty shaker33)
  (empty shot46)
  (empty shot261)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker33 l0)
  (shaker-level shaker33 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail323 ingredient36)
  (cocktail-part2 cocktail323 ingredient234)
)
 (:goal
  (and
      (contains shot46 cocktail323)
)))
