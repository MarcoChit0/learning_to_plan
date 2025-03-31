(define (problem prob)
 (:domain barman)
 (:objects 
      shaker19 - shaker
      left right - hand
      shot164 shot181 shot491 - shot
      ingredient159 ingredient190 ingredient414 ingredient130 - ingredient
      cocktail384 - cocktail
      dispenser129 dispenser427 dispenser363 dispenser152 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker19)
  (ontable shot164)
  (ontable shot181)
  (ontable shot491)
  (dispenses dispenser129 ingredient159)
  (dispenses dispenser427 ingredient190)
  (dispenses dispenser363 ingredient414)
  (dispenses dispenser152 ingredient130)
  (clean shaker19)
  (clean shot164)
  (clean shot181)
  (clean shot491)
  (empty shaker19)
  (empty shot164)
  (empty shot181)
  (empty shot491)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker19 l0)
  (shaker-level shaker19 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail384 ingredient190)
  (cocktail-part2 cocktail384 ingredient130)
)
 (:goal
  (and
      (contains shot164 cocktail384)
      (contains shot181 ingredient159)
)))
