(define (problem prob)
 (:domain barman)
 (:objects 
      shaker356 - shaker
      left right - hand
      shot388 shot56 shot486 - shot
      ingredient377 ingredient302 ingredient207 - ingredient
      cocktail427 - cocktail
      dispenser159 dispenser320 dispenser365 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker356)
  (ontable shot388)
  (ontable shot56)
  (ontable shot486)
  (dispenses dispenser159 ingredient377)
  (dispenses dispenser320 ingredient302)
  (dispenses dispenser365 ingredient207)
  (clean shaker356)
  (clean shot388)
  (clean shot56)
  (clean shot486)
  (empty shaker356)
  (empty shot388)
  (empty shot56)
  (empty shot486)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker356 l0)
  (shaker-level shaker356 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail427 ingredient207)
  (cocktail-part2 cocktail427 ingredient377)
)
 (:goal
  (and
      (contains shot388 cocktail427)
      (contains shot56 cocktail427)
)))
