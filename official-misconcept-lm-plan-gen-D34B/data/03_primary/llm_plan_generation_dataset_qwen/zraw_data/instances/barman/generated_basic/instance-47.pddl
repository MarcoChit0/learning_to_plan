(define (problem prob)
 (:domain barman)
 (:objects 
      shaker241 - shaker
      left right - hand
      shot193 shot263 - shot
      ingredient417 ingredient107 ingredient350 ingredient431 - ingredient
      cocktail1 - cocktail
      dispenser245 dispenser351 dispenser431 dispenser205 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker241)
  (ontable shot193)
  (ontable shot263)
  (dispenses dispenser245 ingredient417)
  (dispenses dispenser351 ingredient107)
  (dispenses dispenser431 ingredient350)
  (dispenses dispenser205 ingredient431)
  (clean shaker241)
  (clean shot193)
  (clean shot263)
  (empty shaker241)
  (empty shot193)
  (empty shot263)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker241 l0)
  (shaker-level shaker241 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient417)
  (cocktail-part2 cocktail1 ingredient350)
)
 (:goal
  (and
      (contains shot193 cocktail1)
)))
