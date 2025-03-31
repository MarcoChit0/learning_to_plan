(define (problem prob)
 (:domain barman)
 (:objects 
      shaker148 - shaker
      left right - hand
      shot196 shot368 shot401 - shot
      ingredient156 ingredient310 ingredient478 - ingredient
      cocktail237 - cocktail
      dispenser31 dispenser470 dispenser405 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker148)
  (ontable shot196)
  (ontable shot368)
  (ontable shot401)
  (dispenses dispenser31 ingredient156)
  (dispenses dispenser470 ingredient310)
  (dispenses dispenser405 ingredient478)
  (clean shaker148)
  (clean shot196)
  (clean shot368)
  (clean shot401)
  (empty shaker148)
  (empty shot196)
  (empty shot368)
  (empty shot401)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker148 l0)
  (shaker-level shaker148 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail237 ingredient156)
  (cocktail-part2 cocktail237 ingredient478)
)
 (:goal
  (and
      (contains shot196 cocktail237)
      (contains shot368 cocktail237)
)))
