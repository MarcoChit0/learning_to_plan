(define (problem prob)
 (:domain barman)
 (:objects 
      shaker320 - shaker
      left right - hand
      shot488 shot337 - shot
      ingredient90 ingredient150 ingredient290 - ingredient
      cocktail137 - cocktail
      dispenser222 dispenser304 dispenser20 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker320)
  (ontable shot488)
  (ontable shot337)
  (dispenses dispenser222 ingredient90)
  (dispenses dispenser304 ingredient150)
  (dispenses dispenser20 ingredient290)
  (clean shaker320)
  (clean shot488)
  (clean shot337)
  (empty shaker320)
  (empty shot488)
  (empty shot337)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker320 l0)
  (shaker-level shaker320 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail137 ingredient90)
  (cocktail-part2 cocktail137 ingredient150)
)
 (:goal
  (and
      (contains shot488 cocktail137)
)))
