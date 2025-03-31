(define (problem prob)
 (:domain barman)
 (:objects 
      shaker459 - shaker
      left right - hand
      shot323 shot330 - shot
      ingredient55 ingredient294 ingredient88 ingredient72 - ingredient
      cocktail116 - cocktail
      dispenser496 dispenser398 dispenser23 dispenser85 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker459)
  (ontable shot323)
  (ontable shot330)
  (dispenses dispenser496 ingredient55)
  (dispenses dispenser398 ingredient294)
  (dispenses dispenser23 ingredient88)
  (dispenses dispenser85 ingredient72)
  (clean shaker459)
  (clean shot323)
  (clean shot330)
  (empty shaker459)
  (empty shot323)
  (empty shot330)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker459 l0)
  (shaker-level shaker459 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail116 ingredient294)
  (cocktail-part2 cocktail116 ingredient88)
)
 (:goal
  (and
      (contains shot323 cocktail116)
)))
