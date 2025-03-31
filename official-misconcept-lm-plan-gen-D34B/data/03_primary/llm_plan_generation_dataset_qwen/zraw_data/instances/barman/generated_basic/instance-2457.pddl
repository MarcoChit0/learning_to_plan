(define (problem prob)
 (:domain barman)
 (:objects 
      shaker227 - shaker
      left right - hand
      shot298 shot104 shot281 - shot
      ingredient332 ingredient2 ingredient10 ingredient415 - ingredient
      cocktail143 - cocktail
      dispenser243 dispenser39 dispenser384 dispenser486 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker227)
  (ontable shot298)
  (ontable shot104)
  (ontable shot281)
  (dispenses dispenser243 ingredient332)
  (dispenses dispenser39 ingredient2)
  (dispenses dispenser384 ingredient10)
  (dispenses dispenser486 ingredient415)
  (clean shaker227)
  (clean shot298)
  (clean shot104)
  (clean shot281)
  (empty shaker227)
  (empty shot298)
  (empty shot104)
  (empty shot281)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker227 l0)
  (shaker-level shaker227 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail143 ingredient2)
  (cocktail-part2 cocktail143 ingredient415)
)
 (:goal
  (and
      (contains shot298 cocktail143)
      (contains shot104 cocktail143)
)))
