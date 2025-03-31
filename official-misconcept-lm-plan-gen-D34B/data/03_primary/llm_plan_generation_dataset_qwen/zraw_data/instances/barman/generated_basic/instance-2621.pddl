(define (problem prob)
 (:domain barman)
 (:objects 
      shaker330 - shaker
      left right - hand
      shot127 shot141 - shot
      ingredient86 ingredient96 ingredient58 - ingredient
      cocktail472 - cocktail
      dispenser140 dispenser62 dispenser150 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker330)
  (ontable shot127)
  (ontable shot141)
  (dispenses dispenser140 ingredient86)
  (dispenses dispenser62 ingredient96)
  (dispenses dispenser150 ingredient58)
  (clean shaker330)
  (clean shot127)
  (clean shot141)
  (empty shaker330)
  (empty shot127)
  (empty shot141)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker330 l0)
  (shaker-level shaker330 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail472 ingredient58)
  (cocktail-part2 cocktail472 ingredient86)
)
 (:goal
  (and
      (contains shot127 cocktail472)
)))
