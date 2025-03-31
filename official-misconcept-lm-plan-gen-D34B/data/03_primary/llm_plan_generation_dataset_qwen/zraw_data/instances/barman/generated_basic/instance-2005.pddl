(define (problem prob)
 (:domain barman)
 (:objects 
      shaker485 - shaker
      left right - hand
      shot33 shot253 shot385 - shot
      ingredient84 ingredient179 ingredient495 - ingredient
      cocktail247 - cocktail
      dispenser260 dispenser383 dispenser449 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker485)
  (ontable shot33)
  (ontable shot253)
  (ontable shot385)
  (dispenses dispenser260 ingredient84)
  (dispenses dispenser383 ingredient179)
  (dispenses dispenser449 ingredient495)
  (clean shaker485)
  (clean shot33)
  (clean shot253)
  (clean shot385)
  (empty shaker485)
  (empty shot33)
  (empty shot253)
  (empty shot385)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker485 l0)
  (shaker-level shaker485 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail247 ingredient179)
  (cocktail-part2 cocktail247 ingredient84)
)
 (:goal
  (and
      (contains shot33 cocktail247)
      (contains shot253 cocktail247)
)))
