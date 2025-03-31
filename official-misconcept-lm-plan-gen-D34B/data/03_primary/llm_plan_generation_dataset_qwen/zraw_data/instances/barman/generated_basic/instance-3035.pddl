(define (problem prob)
 (:domain barman)
 (:objects 
      shaker179 - shaker
      left right - hand
      shot234 - shot
      ingredient467 ingredient4 ingredient170 - ingredient
      cocktail441 - cocktail
      dispenser78 dispenser289 dispenser447 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker179)
  (ontable shot234)
  (dispenses dispenser78 ingredient467)
  (dispenses dispenser289 ingredient4)
  (dispenses dispenser447 ingredient170)
  (clean shaker179)
  (clean shot234)
  (empty shaker179)
  (empty shot234)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker179 l0)
  (shaker-level shaker179 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail441 ingredient467)
  (cocktail-part2 cocktail441 ingredient170)
)
 (:goal
  (and
      (contains shot234 cocktail441)
)))
