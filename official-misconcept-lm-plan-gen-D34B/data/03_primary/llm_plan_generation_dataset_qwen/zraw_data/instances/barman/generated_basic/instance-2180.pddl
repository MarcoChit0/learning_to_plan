(define (problem prob)
 (:domain barman)
 (:objects 
      shaker37 - shaker
      left right - hand
      shot161 shot311 shot6 - shot
      ingredient445 ingredient228 - ingredient
      cocktail137 - cocktail
      dispenser306 dispenser71 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker37)
  (ontable shot161)
  (ontable shot311)
  (ontable shot6)
  (dispenses dispenser306 ingredient445)
  (dispenses dispenser71 ingredient228)
  (clean shaker37)
  (clean shot161)
  (clean shot311)
  (clean shot6)
  (empty shaker37)
  (empty shot161)
  (empty shot311)
  (empty shot6)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker37 l0)
  (shaker-level shaker37 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail137 ingredient228)
  (cocktail-part2 cocktail137 ingredient445)
)
 (:goal
  (and
      (contains shot161 cocktail137)
      (contains shot311 cocktail137)
)))
