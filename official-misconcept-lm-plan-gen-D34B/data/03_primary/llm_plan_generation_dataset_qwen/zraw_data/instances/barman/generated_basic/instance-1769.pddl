(define (problem prob)
 (:domain barman)
 (:objects 
      shaker318 - shaker
      left right - hand
      shot324 shot454 shot234 - shot
      ingredient110 ingredient374 - ingredient
      cocktail244 - cocktail
      dispenser75 dispenser407 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker318)
  (ontable shot324)
  (ontable shot454)
  (ontable shot234)
  (dispenses dispenser75 ingredient110)
  (dispenses dispenser407 ingredient374)
  (clean shaker318)
  (clean shot324)
  (clean shot454)
  (clean shot234)
  (empty shaker318)
  (empty shot324)
  (empty shot454)
  (empty shot234)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker318 l0)
  (shaker-level shaker318 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail244 ingredient110)
  (cocktail-part2 cocktail244 ingredient374)
)
 (:goal
  (and
      (contains shot324 cocktail244)
      (contains shot454 cocktail244)
)))
