(define (problem prob)
 (:domain barman)
 (:objects 
      shaker224 - shaker
      left right - hand
      shot182 shot87 - shot
      ingredient223 ingredient88 - ingredient
      cocktail236 - cocktail
      dispenser98 dispenser402 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker224)
  (ontable shot182)
  (ontable shot87)
  (dispenses dispenser98 ingredient223)
  (dispenses dispenser402 ingredient88)
  (clean shaker224)
  (clean shot182)
  (clean shot87)
  (empty shaker224)
  (empty shot182)
  (empty shot87)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker224 l0)
  (shaker-level shaker224 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail236 ingredient88)
  (cocktail-part2 cocktail236 ingredient223)
)
 (:goal
  (and
      (contains shot182 cocktail236)
)))
