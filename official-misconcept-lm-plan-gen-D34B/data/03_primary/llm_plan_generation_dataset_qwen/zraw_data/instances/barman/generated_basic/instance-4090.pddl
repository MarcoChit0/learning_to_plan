(define (problem prob)
 (:domain barman)
 (:objects 
      shaker244 - shaker
      left right - hand
      shot62 shot190 - shot
      ingredient481 ingredient334 ingredient307 ingredient18 - ingredient
      cocktail93 - cocktail
      dispenser148 dispenser283 dispenser445 dispenser319 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker244)
  (ontable shot62)
  (ontable shot190)
  (dispenses dispenser148 ingredient481)
  (dispenses dispenser283 ingredient334)
  (dispenses dispenser445 ingredient307)
  (dispenses dispenser319 ingredient18)
  (clean shaker244)
  (clean shot62)
  (clean shot190)
  (empty shaker244)
  (empty shot62)
  (empty shot190)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker244 l0)
  (shaker-level shaker244 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail93 ingredient334)
  (cocktail-part2 cocktail93 ingredient307)
)
 (:goal
  (and
      (contains shot62 cocktail93)
)))
