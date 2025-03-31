(define (problem prob)
 (:domain barman)
 (:objects 
      shaker405 - shaker
      left right - hand
      shot8 - shot
      ingredient321 ingredient459 ingredient153 - ingredient
      cocktail198 - cocktail
      dispenser257 dispenser273 dispenser301 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker405)
  (ontable shot8)
  (dispenses dispenser257 ingredient321)
  (dispenses dispenser273 ingredient459)
  (dispenses dispenser301 ingredient153)
  (clean shaker405)
  (clean shot8)
  (empty shaker405)
  (empty shot8)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker405 l0)
  (shaker-level shaker405 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail198 ingredient321)
  (cocktail-part2 cocktail198 ingredient459)
)
 (:goal
  (and
      (contains shot8 cocktail198)
)))
