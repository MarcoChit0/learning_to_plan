(define (problem prob)
 (:domain barman)
 (:objects 
      shaker236 - shaker
      left right - hand
      shot485 - shot
      ingredient207 ingredient51 - ingredient
      cocktail1 - cocktail
      dispenser167 dispenser42 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker236)
  (ontable shot485)
  (dispenses dispenser167 ingredient207)
  (dispenses dispenser42 ingredient51)
  (clean shaker236)
  (clean shot485)
  (empty shaker236)
  (empty shot485)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker236 l0)
  (shaker-level shaker236 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient51)
  (cocktail-part2 cocktail1 ingredient207)
)
 (:goal
  (and
      (contains shot485 cocktail1)
)))
