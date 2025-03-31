(define (problem prob)
 (:domain barman)
 (:objects 
      shaker411 - shaker
      left right - hand
      shot84 shot368 shot318 - shot
      ingredient366 ingredient357 - ingredient
      cocktail288 - cocktail
      dispenser428 dispenser368 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker411)
  (ontable shot84)
  (ontable shot368)
  (ontable shot318)
  (dispenses dispenser428 ingredient366)
  (dispenses dispenser368 ingredient357)
  (clean shaker411)
  (clean shot84)
  (clean shot368)
  (clean shot318)
  (empty shaker411)
  (empty shot84)
  (empty shot368)
  (empty shot318)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker411 l0)
  (shaker-level shaker411 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail288 ingredient357)
  (cocktail-part2 cocktail288 ingredient366)
)
 (:goal
  (and
      (contains shot84 cocktail288)
      (contains shot368 ingredient366)
)))
