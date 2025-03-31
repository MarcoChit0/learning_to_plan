(define (problem prob)
 (:domain barman)
 (:objects 
      shaker426 - shaker
      left right - hand
      shot120 shot490 shot273 - shot
      ingredient142 ingredient354 - ingredient
      cocktail349 - cocktail
      dispenser405 dispenser62 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker426)
  (ontable shot120)
  (ontable shot490)
  (ontable shot273)
  (dispenses dispenser405 ingredient142)
  (dispenses dispenser62 ingredient354)
  (clean shaker426)
  (clean shot120)
  (clean shot490)
  (clean shot273)
  (empty shaker426)
  (empty shot120)
  (empty shot490)
  (empty shot273)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker426 l0)
  (shaker-level shaker426 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail349 ingredient354)
  (cocktail-part2 cocktail349 ingredient142)
)
 (:goal
  (and
      (contains shot120 cocktail349)
      (contains shot490 cocktail349)
)))
