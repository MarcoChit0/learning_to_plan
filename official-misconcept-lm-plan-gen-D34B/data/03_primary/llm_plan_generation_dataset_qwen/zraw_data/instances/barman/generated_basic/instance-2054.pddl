(define (problem prob)
 (:domain barman)
 (:objects 
      shaker461 - shaker
      left right - hand
      shot221 - shot
      ingredient138 ingredient86 ingredient215 ingredient443 - ingredient
      cocktail155 - cocktail
      dispenser306 dispenser185 dispenser12 dispenser276 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker461)
  (ontable shot221)
  (dispenses dispenser306 ingredient138)
  (dispenses dispenser185 ingredient86)
  (dispenses dispenser12 ingredient215)
  (dispenses dispenser276 ingredient443)
  (clean shaker461)
  (clean shot221)
  (empty shaker461)
  (empty shot221)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker461 l0)
  (shaker-level shaker461 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail155 ingredient86)
  (cocktail-part2 cocktail155 ingredient138)
)
 (:goal
  (and
      (contains shot221 cocktail155)
)))
