(define (problem prob)
 (:domain barman)
 (:objects 
      shaker136 - shaker
      left right - hand
      shot473 - shot
      ingredient377 ingredient256 - ingredient
      cocktail36 - cocktail
      dispenser10 dispenser296 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker136)
  (ontable shot473)
  (dispenses dispenser10 ingredient377)
  (dispenses dispenser296 ingredient256)
  (clean shaker136)
  (clean shot473)
  (empty shaker136)
  (empty shot473)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker136 l0)
  (shaker-level shaker136 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail36 ingredient377)
  (cocktail-part2 cocktail36 ingredient256)
)
 (:goal
  (and
      (contains shot473 cocktail36)
)))
