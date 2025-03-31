(define (problem prob)
 (:domain barman)
 (:objects 
      shaker66 - shaker
      left right - hand
      shot486 shot195 - shot
      ingredient239 ingredient387 ingredient283 ingredient23 - ingredient
      cocktail1 - cocktail
      dispenser372 dispenser302 dispenser8 dispenser472 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker66)
  (ontable shot486)
  (ontable shot195)
  (dispenses dispenser372 ingredient239)
  (dispenses dispenser302 ingredient387)
  (dispenses dispenser8 ingredient283)
  (dispenses dispenser472 ingredient23)
  (clean shaker66)
  (clean shot486)
  (clean shot195)
  (empty shaker66)
  (empty shot486)
  (empty shot195)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker66 l0)
  (shaker-level shaker66 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient23)
  (cocktail-part2 cocktail1 ingredient387)
)
 (:goal
  (and
      (contains shot486 cocktail1)
)))
