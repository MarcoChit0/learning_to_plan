(define (problem prob)
 (:domain barman)
 (:objects 
      shaker358 - shaker
      left right - hand
      shot470 shot17 - shot
      ingredient220 ingredient175 ingredient223 ingredient116 - ingredient
      cocktail459 - cocktail
      dispenser264 dispenser232 dispenser103 dispenser63 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker358)
  (ontable shot470)
  (ontable shot17)
  (dispenses dispenser264 ingredient220)
  (dispenses dispenser232 ingredient175)
  (dispenses dispenser103 ingredient223)
  (dispenses dispenser63 ingredient116)
  (clean shaker358)
  (clean shot470)
  (clean shot17)
  (empty shaker358)
  (empty shot470)
  (empty shot17)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker358 l0)
  (shaker-level shaker358 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail459 ingredient116)
  (cocktail-part2 cocktail459 ingredient175)
)
 (:goal
  (and
      (contains shot470 cocktail459)
)))
