(define (problem prob)
 (:domain barman)
 (:objects 
      shaker161 - shaker
      left right - hand
      shot256 shot380 - shot
      ingredient190 ingredient490 ingredient242 - ingredient
      cocktail240 - cocktail
      dispenser216 dispenser39 dispenser118 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker161)
  (ontable shot256)
  (ontable shot380)
  (dispenses dispenser216 ingredient190)
  (dispenses dispenser39 ingredient490)
  (dispenses dispenser118 ingredient242)
  (clean shaker161)
  (clean shot256)
  (clean shot380)
  (empty shaker161)
  (empty shot256)
  (empty shot380)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker161 l0)
  (shaker-level shaker161 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail240 ingredient242)
  (cocktail-part2 cocktail240 ingredient190)
)
 (:goal
  (and
      (contains shot256 cocktail240)
)))
