(define (problem prob)
 (:domain barman)
 (:objects 
      shaker306 - shaker
      left right - hand
      shot453 - shot
      ingredient169 ingredient244 ingredient334 ingredient142 - ingredient
      cocktail231 - cocktail
      dispenser319 dispenser407 dispenser342 dispenser397 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker306)
  (ontable shot453)
  (dispenses dispenser319 ingredient169)
  (dispenses dispenser407 ingredient244)
  (dispenses dispenser342 ingredient334)
  (dispenses dispenser397 ingredient142)
  (clean shaker306)
  (clean shot453)
  (empty shaker306)
  (empty shot453)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker306 l0)
  (shaker-level shaker306 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail231 ingredient244)
  (cocktail-part2 cocktail231 ingredient334)
)
 (:goal
  (and
      (contains shot453 cocktail231)
)))
