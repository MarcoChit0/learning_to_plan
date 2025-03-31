(define (problem prob)
 (:domain barman)
 (:objects 
      shaker370 - shaker
      left right - hand
      shot340 shot79 - shot
      ingredient27 ingredient64 ingredient229 ingredient337 - ingredient
      cocktail494 - cocktail
      dispenser255 dispenser416 dispenser385 dispenser360 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker370)
  (ontable shot340)
  (ontable shot79)
  (dispenses dispenser255 ingredient27)
  (dispenses dispenser416 ingredient64)
  (dispenses dispenser385 ingredient229)
  (dispenses dispenser360 ingredient337)
  (clean shaker370)
  (clean shot340)
  (clean shot79)
  (empty shaker370)
  (empty shot340)
  (empty shot79)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker370 l0)
  (shaker-level shaker370 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail494 ingredient27)
  (cocktail-part2 cocktail494 ingredient64)
)
 (:goal
  (and
      (contains shot340 cocktail494)
)))
