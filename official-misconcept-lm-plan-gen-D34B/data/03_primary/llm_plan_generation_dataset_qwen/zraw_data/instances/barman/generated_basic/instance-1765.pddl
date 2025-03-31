(define (problem prob)
 (:domain barman)
 (:objects 
      shaker434 - shaker
      left right - hand
      shot5 shot193 shot460 - shot
      ingredient94 ingredient490 - ingredient
      cocktail320 - cocktail
      dispenser300 dispenser386 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker434)
  (ontable shot5)
  (ontable shot193)
  (ontable shot460)
  (dispenses dispenser300 ingredient94)
  (dispenses dispenser386 ingredient490)
  (clean shaker434)
  (clean shot5)
  (clean shot193)
  (clean shot460)
  (empty shaker434)
  (empty shot5)
  (empty shot193)
  (empty shot460)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker434 l0)
  (shaker-level shaker434 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail320 ingredient94)
  (cocktail-part2 cocktail320 ingredient490)
)
 (:goal
  (and
      (contains shot5 cocktail320)
      (contains shot193 ingredient490)
)))
