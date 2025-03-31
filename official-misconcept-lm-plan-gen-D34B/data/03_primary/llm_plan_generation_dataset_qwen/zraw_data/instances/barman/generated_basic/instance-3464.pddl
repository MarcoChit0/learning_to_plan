(define (problem prob)
 (:domain barman)
 (:objects 
      shaker443 - shaker
      left right - hand
      shot461 shot269 shot446 - shot
      ingredient423 ingredient469 ingredient405 ingredient395 - ingredient
      cocktail475 - cocktail
      dispenser176 dispenser249 dispenser268 dispenser220 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker443)
  (ontable shot461)
  (ontable shot269)
  (ontable shot446)
  (dispenses dispenser176 ingredient423)
  (dispenses dispenser249 ingredient469)
  (dispenses dispenser268 ingredient405)
  (dispenses dispenser220 ingredient395)
  (clean shaker443)
  (clean shot461)
  (clean shot269)
  (clean shot446)
  (empty shaker443)
  (empty shot461)
  (empty shot269)
  (empty shot446)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker443 l0)
  (shaker-level shaker443 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail475 ingredient405)
  (cocktail-part2 cocktail475 ingredient469)
)
 (:goal
  (and
      (contains shot461 cocktail475)
      (contains shot269 cocktail475)
)))
