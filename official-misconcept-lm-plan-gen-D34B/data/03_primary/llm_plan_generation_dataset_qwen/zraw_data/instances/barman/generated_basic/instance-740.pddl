(define (problem prob)
 (:domain barman)
 (:objects 
      shaker415 - shaker
      left right - hand
      shot370 shot45 - shot
      ingredient474 ingredient288 ingredient224 - ingredient
      cocktail1 - cocktail
      dispenser407 dispenser48 dispenser57 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker415)
  (ontable shot370)
  (ontable shot45)
  (dispenses dispenser407 ingredient474)
  (dispenses dispenser48 ingredient288)
  (dispenses dispenser57 ingredient224)
  (clean shaker415)
  (clean shot370)
  (clean shot45)
  (empty shaker415)
  (empty shot370)
  (empty shot45)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker415 l0)
  (shaker-level shaker415 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient288)
  (cocktail-part2 cocktail1 ingredient224)
)
 (:goal
  (and
      (contains shot370 cocktail1)
)))
