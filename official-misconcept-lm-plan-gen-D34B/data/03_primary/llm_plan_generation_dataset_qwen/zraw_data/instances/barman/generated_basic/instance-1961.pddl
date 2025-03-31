(define (problem prob)
 (:domain barman)
 (:objects 
      shaker65 - shaker
      left right - hand
      shot349 shot108 shot78 - shot
      ingredient408 ingredient480 ingredient354 ingredient333 - ingredient
      cocktail373 - cocktail
      dispenser136 dispenser1 dispenser474 dispenser114 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker65)
  (ontable shot349)
  (ontable shot108)
  (ontable shot78)
  (dispenses dispenser136 ingredient408)
  (dispenses dispenser1 ingredient480)
  (dispenses dispenser474 ingredient354)
  (dispenses dispenser114 ingredient333)
  (clean shaker65)
  (clean shot349)
  (clean shot108)
  (clean shot78)
  (empty shaker65)
  (empty shot349)
  (empty shot108)
  (empty shot78)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker65 l0)
  (shaker-level shaker65 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail373 ingredient408)
  (cocktail-part2 cocktail373 ingredient354)
)
 (:goal
  (and
      (contains shot349 cocktail373)
      (contains shot108 cocktail373)
)))
