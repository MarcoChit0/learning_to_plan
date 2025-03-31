(define (problem prob)
 (:domain barman)
 (:objects 
      shaker111 - shaker
      left right - hand
      shot358 shot356 shot455 - shot
      ingredient260 ingredient306 ingredient162 ingredient94 - ingredient
      cocktail37 - cocktail
      dispenser1 dispenser466 dispenser300 dispenser289 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker111)
  (ontable shot358)
  (ontable shot356)
  (ontable shot455)
  (dispenses dispenser1 ingredient260)
  (dispenses dispenser466 ingredient306)
  (dispenses dispenser300 ingredient162)
  (dispenses dispenser289 ingredient94)
  (clean shaker111)
  (clean shot358)
  (clean shot356)
  (clean shot455)
  (empty shaker111)
  (empty shot358)
  (empty shot356)
  (empty shot455)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker111 l0)
  (shaker-level shaker111 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail37 ingredient94)
  (cocktail-part2 cocktail37 ingredient306)
)
 (:goal
  (and
      (contains shot358 cocktail37)
      (contains shot356 ingredient94)
)))
