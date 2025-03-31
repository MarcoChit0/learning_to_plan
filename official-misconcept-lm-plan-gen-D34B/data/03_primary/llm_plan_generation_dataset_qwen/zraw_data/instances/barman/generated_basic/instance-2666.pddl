(define (problem prob)
 (:domain barman)
 (:objects 
      shaker58 - shaker
      left right - hand
      shot67 - shot
      ingredient203 ingredient358 ingredient122 ingredient488 - ingredient
      cocktail272 - cocktail
      dispenser392 dispenser254 dispenser351 dispenser457 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker58)
  (ontable shot67)
  (dispenses dispenser392 ingredient203)
  (dispenses dispenser254 ingredient358)
  (dispenses dispenser351 ingredient122)
  (dispenses dispenser457 ingredient488)
  (clean shaker58)
  (clean shot67)
  (empty shaker58)
  (empty shot67)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker58 l0)
  (shaker-level shaker58 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail272 ingredient358)
  (cocktail-part2 cocktail272 ingredient203)
)
 (:goal
  (and
      (contains shot67 cocktail272)
)))
