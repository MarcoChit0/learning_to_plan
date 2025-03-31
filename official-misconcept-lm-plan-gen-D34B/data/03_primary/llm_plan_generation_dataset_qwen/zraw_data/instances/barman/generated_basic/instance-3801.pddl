(define (problem prob)
 (:domain barman)
 (:objects 
      shaker177 - shaker
      left right - hand
      shot486 - shot
      ingredient449 ingredient46 ingredient162 ingredient215 - ingredient
      cocktail453 - cocktail
      dispenser316 dispenser155 dispenser157 dispenser140 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker177)
  (ontable shot486)
  (dispenses dispenser316 ingredient449)
  (dispenses dispenser155 ingredient46)
  (dispenses dispenser157 ingredient162)
  (dispenses dispenser140 ingredient215)
  (clean shaker177)
  (clean shot486)
  (empty shaker177)
  (empty shot486)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker177 l0)
  (shaker-level shaker177 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail453 ingredient215)
  (cocktail-part2 cocktail453 ingredient162)
)
 (:goal
  (and
      (contains shot486 cocktail453)
)))
