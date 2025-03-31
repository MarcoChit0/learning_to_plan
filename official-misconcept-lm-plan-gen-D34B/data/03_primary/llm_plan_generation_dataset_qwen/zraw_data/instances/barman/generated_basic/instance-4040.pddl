(define (problem prob)
 (:domain barman)
 (:objects 
      shaker119 - shaker
      left right - hand
      shot234 shot189 - shot
      ingredient348 ingredient314 ingredient24 ingredient267 - ingredient
      cocktail107 - cocktail
      dispenser149 dispenser127 dispenser366 dispenser308 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker119)
  (ontable shot234)
  (ontable shot189)
  (dispenses dispenser149 ingredient348)
  (dispenses dispenser127 ingredient314)
  (dispenses dispenser366 ingredient24)
  (dispenses dispenser308 ingredient267)
  (clean shaker119)
  (clean shot234)
  (clean shot189)
  (empty shaker119)
  (empty shot234)
  (empty shot189)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker119 l0)
  (shaker-level shaker119 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail107 ingredient348)
  (cocktail-part2 cocktail107 ingredient24)
)
 (:goal
  (and
      (contains shot234 cocktail107)
)))
