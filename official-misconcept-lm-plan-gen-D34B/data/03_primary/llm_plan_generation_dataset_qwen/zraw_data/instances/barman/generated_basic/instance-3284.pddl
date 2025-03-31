(define (problem prob)
 (:domain barman)
 (:objects 
      shaker201 - shaker
      left right - hand
      shot112 shot350 - shot
      ingredient269 ingredient347 ingredient37 - ingredient
      cocktail138 - cocktail
      dispenser317 dispenser255 dispenser166 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker201)
  (ontable shot112)
  (ontable shot350)
  (dispenses dispenser317 ingredient269)
  (dispenses dispenser255 ingredient347)
  (dispenses dispenser166 ingredient37)
  (clean shaker201)
  (clean shot112)
  (clean shot350)
  (empty shaker201)
  (empty shot112)
  (empty shot350)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker201 l0)
  (shaker-level shaker201 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail138 ingredient37)
  (cocktail-part2 cocktail138 ingredient269)
)
 (:goal
  (and
      (contains shot112 cocktail138)
)))
