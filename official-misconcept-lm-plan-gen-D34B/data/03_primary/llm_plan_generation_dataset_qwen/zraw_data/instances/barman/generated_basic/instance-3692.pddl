(define (problem prob)
 (:domain barman)
 (:objects 
      shaker244 - shaker
      left right - hand
      shot252 shot320 - shot
      ingredient201 ingredient97 ingredient51 - ingredient
      cocktail68 - cocktail
      dispenser413 dispenser253 dispenser235 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker244)
  (ontable shot252)
  (ontable shot320)
  (dispenses dispenser413 ingredient201)
  (dispenses dispenser253 ingredient97)
  (dispenses dispenser235 ingredient51)
  (clean shaker244)
  (clean shot252)
  (clean shot320)
  (empty shaker244)
  (empty shot252)
  (empty shot320)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker244 l0)
  (shaker-level shaker244 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail68 ingredient201)
  (cocktail-part2 cocktail68 ingredient97)
)
 (:goal
  (and
      (contains shot252 cocktail68)
)))
