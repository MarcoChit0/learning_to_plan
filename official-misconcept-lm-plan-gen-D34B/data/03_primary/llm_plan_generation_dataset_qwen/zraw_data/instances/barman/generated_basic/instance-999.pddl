(define (problem prob)
 (:domain barman)
 (:objects 
      shaker305 - shaker
      left right - hand
      shot205 shot273 - shot
      ingredient98 ingredient361 - ingredient
      cocktail1 - cocktail
      dispenser155 dispenser378 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker305)
  (ontable shot205)
  (ontable shot273)
  (dispenses dispenser155 ingredient98)
  (dispenses dispenser378 ingredient361)
  (clean shaker305)
  (clean shot205)
  (clean shot273)
  (empty shaker305)
  (empty shot205)
  (empty shot273)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker305 l0)
  (shaker-level shaker305 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient361)
  (cocktail-part2 cocktail1 ingredient98)
)
 (:goal
  (and
      (contains shot205 cocktail1)
)))
