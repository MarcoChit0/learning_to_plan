(define (problem prob)
 (:domain barman)
 (:objects 
      shaker75 - shaker
      left right - hand
      shot338 shot127 - shot
      ingredient466 ingredient131 - ingredient
      cocktail1 - cocktail
      dispenser447 dispenser424 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker75)
  (ontable shot338)
  (ontable shot127)
  (dispenses dispenser447 ingredient466)
  (dispenses dispenser424 ingredient131)
  (clean shaker75)
  (clean shot338)
  (clean shot127)
  (empty shaker75)
  (empty shot338)
  (empty shot127)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker75 l0)
  (shaker-level shaker75 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient131)
  (cocktail-part2 cocktail1 ingredient466)
)
 (:goal
  (and
      (contains shot338 cocktail1)
)))
