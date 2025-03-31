(define (problem prob)
 (:domain barman)
 (:objects 
      shaker328 - shaker
      left right - hand
      shot355 - shot
      ingredient176 ingredient97 ingredient233 ingredient8 - ingredient
      cocktail1 - cocktail
      dispenser44 dispenser382 dispenser239 dispenser223 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker328)
  (ontable shot355)
  (dispenses dispenser44 ingredient176)
  (dispenses dispenser382 ingredient97)
  (dispenses dispenser239 ingredient233)
  (dispenses dispenser223 ingredient8)
  (clean shaker328)
  (clean shot355)
  (empty shaker328)
  (empty shot355)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker328 l0)
  (shaker-level shaker328 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient8)
  (cocktail-part2 cocktail1 ingredient97)
)
 (:goal
  (and
      (contains shot355 cocktail1)
)))
