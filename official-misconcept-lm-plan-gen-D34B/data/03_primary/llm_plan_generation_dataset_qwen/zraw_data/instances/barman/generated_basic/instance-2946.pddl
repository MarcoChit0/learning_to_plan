(define (problem prob)
 (:domain barman)
 (:objects 
      shaker379 - shaker
      left right - hand
      shot144 - shot
      ingredient103 ingredient168 ingredient143 ingredient386 - ingredient
      cocktail27 - cocktail
      dispenser259 dispenser364 dispenser408 dispenser293 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker379)
  (ontable shot144)
  (dispenses dispenser259 ingredient103)
  (dispenses dispenser364 ingredient168)
  (dispenses dispenser408 ingredient143)
  (dispenses dispenser293 ingredient386)
  (clean shaker379)
  (clean shot144)
  (empty shaker379)
  (empty shot144)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker379 l0)
  (shaker-level shaker379 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail27 ingredient143)
  (cocktail-part2 cocktail27 ingredient168)
)
 (:goal
  (and
      (contains shot144 cocktail27)
)))
