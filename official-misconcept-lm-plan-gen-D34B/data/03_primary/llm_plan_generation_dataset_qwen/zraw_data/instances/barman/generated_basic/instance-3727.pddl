(define (problem prob)
 (:domain barman)
 (:objects 
      shaker32 - shaker
      left right - hand
      shot94 - shot
      ingredient404 ingredient10 ingredient269 ingredient168 - ingredient
      cocktail460 - cocktail
      dispenser319 dispenser332 dispenser37 dispenser431 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker32)
  (ontable shot94)
  (dispenses dispenser319 ingredient404)
  (dispenses dispenser332 ingredient10)
  (dispenses dispenser37 ingredient269)
  (dispenses dispenser431 ingredient168)
  (clean shaker32)
  (clean shot94)
  (empty shaker32)
  (empty shot94)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker32 l0)
  (shaker-level shaker32 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail460 ingredient404)
  (cocktail-part2 cocktail460 ingredient10)
)
 (:goal
  (and
      (contains shot94 cocktail460)
)))
