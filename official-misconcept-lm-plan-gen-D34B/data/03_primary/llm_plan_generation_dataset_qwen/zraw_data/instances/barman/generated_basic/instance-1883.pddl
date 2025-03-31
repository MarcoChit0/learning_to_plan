(define (problem prob)
 (:domain barman)
 (:objects 
      shaker91 - shaker
      left right - hand
      shot479 shot486 - shot
      ingredient382 ingredient323 ingredient11 - ingredient
      cocktail296 - cocktail
      dispenser392 dispenser168 dispenser96 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker91)
  (ontable shot479)
  (ontable shot486)
  (dispenses dispenser392 ingredient382)
  (dispenses dispenser168 ingredient323)
  (dispenses dispenser96 ingredient11)
  (clean shaker91)
  (clean shot479)
  (clean shot486)
  (empty shaker91)
  (empty shot479)
  (empty shot486)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker91 l0)
  (shaker-level shaker91 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail296 ingredient11)
  (cocktail-part2 cocktail296 ingredient382)
)
 (:goal
  (and
      (contains shot479 cocktail296)
)))
