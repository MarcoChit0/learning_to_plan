(define (problem prob)
 (:domain barman)
 (:objects 
      shaker318 - shaker
      left right - hand
      shot207 shot140 - shot
      ingredient8 ingredient295 - ingredient
      cocktail1 - cocktail
      dispenser343 dispenser112 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker318)
  (ontable shot207)
  (ontable shot140)
  (dispenses dispenser343 ingredient8)
  (dispenses dispenser112 ingredient295)
  (clean shaker318)
  (clean shot207)
  (clean shot140)
  (empty shaker318)
  (empty shot207)
  (empty shot140)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker318 l0)
  (shaker-level shaker318 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient295)
  (cocktail-part2 cocktail1 ingredient8)
)
 (:goal
  (and
      (contains shot207 cocktail1)
)))
