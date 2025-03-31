(define (problem prob)
 (:domain barman)
 (:objects 
      shaker173 - shaker
      left right - hand
      shot63 shot234 - shot
      ingredient195 ingredient136 - ingredient
      cocktail79 - cocktail
      dispenser368 dispenser70 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker173)
  (ontable shot63)
  (ontable shot234)
  (dispenses dispenser368 ingredient195)
  (dispenses dispenser70 ingredient136)
  (clean shaker173)
  (clean shot63)
  (clean shot234)
  (empty shaker173)
  (empty shot63)
  (empty shot234)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker173 l0)
  (shaker-level shaker173 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail79 ingredient136)
  (cocktail-part2 cocktail79 ingredient195)
)
 (:goal
  (and
      (contains shot63 cocktail79)
)))
