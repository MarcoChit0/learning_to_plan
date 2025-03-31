(define (problem prob)
 (:domain barman)
 (:objects 
      shaker384 - shaker
      left right - hand
      shot0 shot478 - shot
      ingredient301 ingredient233 ingredient477 ingredient476 - ingredient
      cocktail6 - cocktail
      dispenser135 dispenser394 dispenser172 dispenser431 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker384)
  (ontable shot0)
  (ontable shot478)
  (dispenses dispenser135 ingredient301)
  (dispenses dispenser394 ingredient233)
  (dispenses dispenser172 ingredient477)
  (dispenses dispenser431 ingredient476)
  (clean shaker384)
  (clean shot0)
  (clean shot478)
  (empty shaker384)
  (empty shot0)
  (empty shot478)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker384 l0)
  (shaker-level shaker384 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail6 ingredient301)
  (cocktail-part2 cocktail6 ingredient233)
)
 (:goal
  (and
      (contains shot0 cocktail6)
)))
