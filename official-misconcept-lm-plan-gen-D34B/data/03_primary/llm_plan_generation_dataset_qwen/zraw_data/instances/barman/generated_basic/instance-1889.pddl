(define (problem prob)
 (:domain barman)
 (:objects 
      shaker98 - shaker
      left right - hand
      shot441 shot124 - shot
      ingredient422 ingredient275 ingredient169 - ingredient
      cocktail316 - cocktail
      dispenser465 dispenser4 dispenser337 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker98)
  (ontable shot441)
  (ontable shot124)
  (dispenses dispenser465 ingredient422)
  (dispenses dispenser4 ingredient275)
  (dispenses dispenser337 ingredient169)
  (clean shaker98)
  (clean shot441)
  (clean shot124)
  (empty shaker98)
  (empty shot441)
  (empty shot124)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker98 l0)
  (shaker-level shaker98 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail316 ingredient275)
  (cocktail-part2 cocktail316 ingredient169)
)
 (:goal
  (and
      (contains shot441 cocktail316)
)))
