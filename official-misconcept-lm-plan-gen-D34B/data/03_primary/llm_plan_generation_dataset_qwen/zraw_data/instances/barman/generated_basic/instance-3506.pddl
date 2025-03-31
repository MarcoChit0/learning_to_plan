(define (problem prob)
 (:domain barman)
 (:objects 
      shaker66 - shaker
      left right - hand
      shot110 shot446 - shot
      ingredient342 ingredient195 ingredient436 ingredient27 - ingredient
      cocktail317 - cocktail
      dispenser249 dispenser494 dispenser483 dispenser43 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker66)
  (ontable shot110)
  (ontable shot446)
  (dispenses dispenser249 ingredient342)
  (dispenses dispenser494 ingredient195)
  (dispenses dispenser483 ingredient436)
  (dispenses dispenser43 ingredient27)
  (clean shaker66)
  (clean shot110)
  (clean shot446)
  (empty shaker66)
  (empty shot110)
  (empty shot446)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker66 l0)
  (shaker-level shaker66 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail317 ingredient195)
  (cocktail-part2 cocktail317 ingredient342)
)
 (:goal
  (and
      (contains shot110 cocktail317)
)))
