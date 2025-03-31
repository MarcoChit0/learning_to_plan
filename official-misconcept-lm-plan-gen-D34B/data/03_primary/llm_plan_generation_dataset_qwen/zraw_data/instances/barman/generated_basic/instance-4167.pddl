(define (problem prob)
 (:domain barman)
 (:objects 
      shaker468 - shaker
      left right - hand
      shot203 shot416 - shot
      ingredient155 ingredient393 - ingredient
      cocktail15 - cocktail
      dispenser123 dispenser349 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker468)
  (ontable shot203)
  (ontable shot416)
  (dispenses dispenser123 ingredient155)
  (dispenses dispenser349 ingredient393)
  (clean shaker468)
  (clean shot203)
  (clean shot416)
  (empty shaker468)
  (empty shot203)
  (empty shot416)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker468 l0)
  (shaker-level shaker468 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail15 ingredient155)
  (cocktail-part2 cocktail15 ingredient393)
)
 (:goal
  (and
      (contains shot203 cocktail15)
)))
