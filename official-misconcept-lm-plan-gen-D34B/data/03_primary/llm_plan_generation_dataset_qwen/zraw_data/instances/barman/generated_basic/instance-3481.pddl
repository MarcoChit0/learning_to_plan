(define (problem prob)
 (:domain barman)
 (:objects 
      shaker462 - shaker
      left right - hand
      shot214 shot0 shot423 - shot
      ingredient357 ingredient410 ingredient427 - ingredient
      cocktail95 - cocktail
      dispenser67 dispenser95 dispenser342 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker462)
  (ontable shot214)
  (ontable shot0)
  (ontable shot423)
  (dispenses dispenser67 ingredient357)
  (dispenses dispenser95 ingredient410)
  (dispenses dispenser342 ingredient427)
  (clean shaker462)
  (clean shot214)
  (clean shot0)
  (clean shot423)
  (empty shaker462)
  (empty shot214)
  (empty shot0)
  (empty shot423)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker462 l0)
  (shaker-level shaker462 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail95 ingredient427)
  (cocktail-part2 cocktail95 ingredient357)
)
 (:goal
  (and
      (contains shot214 cocktail95)
      (contains shot0 ingredient410)
)))
