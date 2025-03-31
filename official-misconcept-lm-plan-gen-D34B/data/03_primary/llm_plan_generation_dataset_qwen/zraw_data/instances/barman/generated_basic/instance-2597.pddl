(define (problem prob)
 (:domain barman)
 (:objects 
      shaker388 - shaker
      left right - hand
      shot11 shot249 - shot
      ingredient336 ingredient294 ingredient257 - ingredient
      cocktail180 - cocktail
      dispenser232 dispenser183 dispenser28 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker388)
  (ontable shot11)
  (ontable shot249)
  (dispenses dispenser232 ingredient336)
  (dispenses dispenser183 ingredient294)
  (dispenses dispenser28 ingredient257)
  (clean shaker388)
  (clean shot11)
  (clean shot249)
  (empty shaker388)
  (empty shot11)
  (empty shot249)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker388 l0)
  (shaker-level shaker388 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail180 ingredient294)
  (cocktail-part2 cocktail180 ingredient336)
)
 (:goal
  (and
      (contains shot11 cocktail180)
)))
