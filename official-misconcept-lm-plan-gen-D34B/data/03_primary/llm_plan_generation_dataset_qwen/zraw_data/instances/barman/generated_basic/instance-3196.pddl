(define (problem prob)
 (:domain barman)
 (:objects 
      shaker287 - shaker
      left right - hand
      shot285 shot363 - shot
      ingredient220 ingredient0 ingredient356 - ingredient
      cocktail452 - cocktail
      dispenser168 dispenser229 dispenser188 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker287)
  (ontable shot285)
  (ontable shot363)
  (dispenses dispenser168 ingredient220)
  (dispenses dispenser229 ingredient0)
  (dispenses dispenser188 ingredient356)
  (clean shaker287)
  (clean shot285)
  (clean shot363)
  (empty shaker287)
  (empty shot285)
  (empty shot363)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker287 l0)
  (shaker-level shaker287 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail452 ingredient356)
  (cocktail-part2 cocktail452 ingredient220)
)
 (:goal
  (and
      (contains shot285 cocktail452)
)))
