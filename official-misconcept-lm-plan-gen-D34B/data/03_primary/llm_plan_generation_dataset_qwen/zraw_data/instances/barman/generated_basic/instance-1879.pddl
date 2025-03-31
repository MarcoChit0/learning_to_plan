(define (problem prob)
 (:domain barman)
 (:objects 
      shaker146 - shaker
      left right - hand
      shot24 shot270 - shot
      ingredient142 ingredient147 ingredient268 - ingredient
      cocktail357 - cocktail
      dispenser388 dispenser193 dispenser418 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker146)
  (ontable shot24)
  (ontable shot270)
  (dispenses dispenser388 ingredient142)
  (dispenses dispenser193 ingredient147)
  (dispenses dispenser418 ingredient268)
  (clean shaker146)
  (clean shot24)
  (clean shot270)
  (empty shaker146)
  (empty shot24)
  (empty shot270)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker146 l0)
  (shaker-level shaker146 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail357 ingredient268)
  (cocktail-part2 cocktail357 ingredient142)
)
 (:goal
  (and
      (contains shot24 cocktail357)
)))
