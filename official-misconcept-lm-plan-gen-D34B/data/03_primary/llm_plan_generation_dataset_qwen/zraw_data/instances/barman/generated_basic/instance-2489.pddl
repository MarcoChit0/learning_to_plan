(define (problem prob)
 (:domain barman)
 (:objects 
      shaker36 - shaker
      left right - hand
      shot368 shot394 shot453 - shot
      ingredient183 ingredient170 ingredient336 - ingredient
      cocktail199 - cocktail
      dispenser429 dispenser456 dispenser428 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker36)
  (ontable shot368)
  (ontable shot394)
  (ontable shot453)
  (dispenses dispenser429 ingredient183)
  (dispenses dispenser456 ingredient170)
  (dispenses dispenser428 ingredient336)
  (clean shaker36)
  (clean shot368)
  (clean shot394)
  (clean shot453)
  (empty shaker36)
  (empty shot368)
  (empty shot394)
  (empty shot453)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker36 l0)
  (shaker-level shaker36 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail199 ingredient336)
  (cocktail-part2 cocktail199 ingredient183)
)
 (:goal
  (and
      (contains shot368 cocktail199)
      (contains shot394 ingredient336)
)))
