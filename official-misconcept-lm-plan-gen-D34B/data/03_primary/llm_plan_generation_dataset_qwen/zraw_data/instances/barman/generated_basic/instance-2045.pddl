(define (problem prob)
 (:domain barman)
 (:objects 
      shaker437 - shaker
      left right - hand
      shot492 - shot
      ingredient103 ingredient362 ingredient34 ingredient39 - ingredient
      cocktail190 - cocktail
      dispenser399 dispenser471 dispenser80 dispenser400 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker437)
  (ontable shot492)
  (dispenses dispenser399 ingredient103)
  (dispenses dispenser471 ingredient362)
  (dispenses dispenser80 ingredient34)
  (dispenses dispenser400 ingredient39)
  (clean shaker437)
  (clean shot492)
  (empty shaker437)
  (empty shot492)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker437 l0)
  (shaker-level shaker437 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail190 ingredient34)
  (cocktail-part2 cocktail190 ingredient362)
)
 (:goal
  (and
      (contains shot492 cocktail190)
)))
