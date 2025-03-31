(define (problem prob)
 (:domain barman)
 (:objects 
      shaker76 - shaker
      left right - hand
      shot395 shot317 shot311 - shot
      ingredient25 ingredient460 ingredient439 ingredient294 - ingredient
      cocktail429 - cocktail
      dispenser266 dispenser491 dispenser15 dispenser477 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker76)
  (ontable shot395)
  (ontable shot317)
  (ontable shot311)
  (dispenses dispenser266 ingredient25)
  (dispenses dispenser491 ingredient460)
  (dispenses dispenser15 ingredient439)
  (dispenses dispenser477 ingredient294)
  (clean shaker76)
  (clean shot395)
  (clean shot317)
  (clean shot311)
  (empty shaker76)
  (empty shot395)
  (empty shot317)
  (empty shot311)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker76 l0)
  (shaker-level shaker76 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail429 ingredient294)
  (cocktail-part2 cocktail429 ingredient439)
)
 (:goal
  (and
      (contains shot395 cocktail429)
      (contains shot317 ingredient460)
)))
