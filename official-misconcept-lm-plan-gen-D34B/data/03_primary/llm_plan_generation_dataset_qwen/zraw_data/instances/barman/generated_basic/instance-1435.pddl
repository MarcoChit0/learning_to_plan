(define (problem prob)
 (:domain barman)
 (:objects 
      shaker193 - shaker
      left right - hand
      shot436 - shot
      ingredient132 ingredient19 ingredient300 - ingredient
      cocktail29 - cocktail
      dispenser244 dispenser7 dispenser285 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker193)
  (ontable shot436)
  (dispenses dispenser244 ingredient132)
  (dispenses dispenser7 ingredient19)
  (dispenses dispenser285 ingredient300)
  (clean shaker193)
  (clean shot436)
  (empty shaker193)
  (empty shot436)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker193 l0)
  (shaker-level shaker193 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail29 ingredient19)
  (cocktail-part2 cocktail29 ingredient300)
)
 (:goal
  (and
      (contains shot436 cocktail29)
)))
