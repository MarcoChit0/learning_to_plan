(define (problem prob)
 (:domain barman)
 (:objects 
      shaker395 - shaker
      left right - hand
      shot319 shot156 - shot
      ingredient412 ingredient456 ingredient23 ingredient56 - ingredient
      cocktail92 - cocktail
      dispenser296 dispenser423 dispenser382 dispenser426 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker395)
  (ontable shot319)
  (ontable shot156)
  (dispenses dispenser296 ingredient412)
  (dispenses dispenser423 ingredient456)
  (dispenses dispenser382 ingredient23)
  (dispenses dispenser426 ingredient56)
  (clean shaker395)
  (clean shot319)
  (clean shot156)
  (empty shaker395)
  (empty shot319)
  (empty shot156)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker395 l0)
  (shaker-level shaker395 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail92 ingredient56)
  (cocktail-part2 cocktail92 ingredient456)
)
 (:goal
  (and
      (contains shot319 cocktail92)
)))
