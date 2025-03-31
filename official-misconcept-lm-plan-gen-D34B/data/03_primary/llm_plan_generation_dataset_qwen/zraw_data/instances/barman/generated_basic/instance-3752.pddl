(define (problem prob)
 (:domain barman)
 (:objects 
      shaker109 - shaker
      left right - hand
      shot160 shot402 shot77 - shot
      ingredient220 ingredient439 - ingredient
      cocktail222 - cocktail
      dispenser293 dispenser150 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker109)
  (ontable shot160)
  (ontable shot402)
  (ontable shot77)
  (dispenses dispenser293 ingredient220)
  (dispenses dispenser150 ingredient439)
  (clean shaker109)
  (clean shot160)
  (clean shot402)
  (clean shot77)
  (empty shaker109)
  (empty shot160)
  (empty shot402)
  (empty shot77)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker109 l0)
  (shaker-level shaker109 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail222 ingredient220)
  (cocktail-part2 cocktail222 ingredient439)
)
 (:goal
  (and
      (contains shot160 cocktail222)
      (contains shot402 cocktail222)
)))
