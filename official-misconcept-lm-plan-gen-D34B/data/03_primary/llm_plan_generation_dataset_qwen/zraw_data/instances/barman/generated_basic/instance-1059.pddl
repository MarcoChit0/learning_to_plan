(define (problem prob)
 (:domain barman)
 (:objects 
      shaker32 - shaker
      left right - hand
      shot156 shot431 shot9 - shot
      ingredient49 ingredient211 ingredient88 ingredient60 - ingredient
      cocktail1 - cocktail
      dispenser474 dispenser454 dispenser324 dispenser41 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker32)
  (ontable shot156)
  (ontable shot431)
  (ontable shot9)
  (dispenses dispenser474 ingredient49)
  (dispenses dispenser454 ingredient211)
  (dispenses dispenser324 ingredient88)
  (dispenses dispenser41 ingredient60)
  (clean shaker32)
  (clean shot156)
  (clean shot431)
  (clean shot9)
  (empty shaker32)
  (empty shot156)
  (empty shot431)
  (empty shot9)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker32 l0)
  (shaker-level shaker32 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient211)
  (cocktail-part2 cocktail1 ingredient49)
)
 (:goal
  (and
      (contains shot156 cocktail1)
      (contains shot431 cocktail1)
)))
