(define (problem prob)
 (:domain barman)
 (:objects 
      shaker231 - shaker
      left right - hand
      shot471 shot65 - shot
      ingredient91 ingredient225 ingredient9 - ingredient
      cocktail446 - cocktail
      dispenser80 dispenser54 dispenser37 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker231)
  (ontable shot471)
  (ontable shot65)
  (dispenses dispenser80 ingredient91)
  (dispenses dispenser54 ingredient225)
  (dispenses dispenser37 ingredient9)
  (clean shaker231)
  (clean shot471)
  (clean shot65)
  (empty shaker231)
  (empty shot471)
  (empty shot65)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker231 l0)
  (shaker-level shaker231 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail446 ingredient9)
  (cocktail-part2 cocktail446 ingredient225)
)
 (:goal
  (and
      (contains shot471 cocktail446)
)))
