(define (problem prob)
 (:domain barman)
 (:objects 
      shaker160 - shaker
      left right - hand
      shot388 shot30 - shot
      ingredient303 ingredient112 ingredient80 - ingredient
      cocktail240 - cocktail
      dispenser68 dispenser69 dispenser309 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker160)
  (ontable shot388)
  (ontable shot30)
  (dispenses dispenser68 ingredient303)
  (dispenses dispenser69 ingredient112)
  (dispenses dispenser309 ingredient80)
  (clean shaker160)
  (clean shot388)
  (clean shot30)
  (empty shaker160)
  (empty shot388)
  (empty shot30)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker160 l0)
  (shaker-level shaker160 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail240 ingredient80)
  (cocktail-part2 cocktail240 ingredient112)
)
 (:goal
  (and
      (contains shot388 cocktail240)
)))
