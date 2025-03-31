(define (problem prob)
 (:domain barman)
 (:objects 
      shaker178 - shaker
      left right - hand
      shot101 shot87 shot32 - shot
      ingredient403 ingredient303 ingredient215 ingredient244 - ingredient
      cocktail435 - cocktail
      dispenser145 dispenser182 dispenser66 dispenser431 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker178)
  (ontable shot101)
  (ontable shot87)
  (ontable shot32)
  (dispenses dispenser145 ingredient403)
  (dispenses dispenser182 ingredient303)
  (dispenses dispenser66 ingredient215)
  (dispenses dispenser431 ingredient244)
  (clean shaker178)
  (clean shot101)
  (clean shot87)
  (clean shot32)
  (empty shaker178)
  (empty shot101)
  (empty shot87)
  (empty shot32)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker178 l0)
  (shaker-level shaker178 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail435 ingredient215)
  (cocktail-part2 cocktail435 ingredient244)
)
 (:goal
  (and
      (contains shot101 cocktail435)
      (contains shot87 cocktail435)
)))
