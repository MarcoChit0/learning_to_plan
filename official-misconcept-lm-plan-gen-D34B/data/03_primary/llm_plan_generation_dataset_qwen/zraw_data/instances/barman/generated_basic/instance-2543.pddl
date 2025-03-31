(define (problem prob)
 (:domain barman)
 (:objects 
      shaker276 - shaker
      left right - hand
      shot188 shot368 - shot
      ingredient43 ingredient268 - ingredient
      cocktail271 - cocktail
      dispenser474 dispenser283 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker276)
  (ontable shot188)
  (ontable shot368)
  (dispenses dispenser474 ingredient43)
  (dispenses dispenser283 ingredient268)
  (clean shaker276)
  (clean shot188)
  (clean shot368)
  (empty shaker276)
  (empty shot188)
  (empty shot368)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker276 l0)
  (shaker-level shaker276 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail271 ingredient43)
  (cocktail-part2 cocktail271 ingredient268)
)
 (:goal
  (and
      (contains shot188 cocktail271)
)))
