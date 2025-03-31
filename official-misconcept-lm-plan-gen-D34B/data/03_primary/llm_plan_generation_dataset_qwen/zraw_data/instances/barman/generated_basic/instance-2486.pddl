(define (problem prob)
 (:domain barman)
 (:objects 
      shaker373 - shaker
      left right - hand
      shot249 shot399 shot357 - shot
      ingredient488 ingredient300 ingredient125 - ingredient
      cocktail155 - cocktail
      dispenser86 dispenser65 dispenser301 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker373)
  (ontable shot249)
  (ontable shot399)
  (ontable shot357)
  (dispenses dispenser86 ingredient488)
  (dispenses dispenser65 ingredient300)
  (dispenses dispenser301 ingredient125)
  (clean shaker373)
  (clean shot249)
  (clean shot399)
  (clean shot357)
  (empty shaker373)
  (empty shot249)
  (empty shot399)
  (empty shot357)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker373 l0)
  (shaker-level shaker373 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail155 ingredient300)
  (cocktail-part2 cocktail155 ingredient488)
)
 (:goal
  (and
      (contains shot249 cocktail155)
      (contains shot399 cocktail155)
)))
