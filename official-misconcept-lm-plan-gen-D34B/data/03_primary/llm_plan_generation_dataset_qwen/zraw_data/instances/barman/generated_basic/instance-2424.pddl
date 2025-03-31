(define (problem prob)
 (:domain barman)
 (:objects 
      shaker358 - shaker
      left right - hand
      shot106 shot311 shot168 - shot
      ingredient287 ingredient251 ingredient88 - ingredient
      cocktail258 - cocktail
      dispenser422 dispenser462 dispenser249 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker358)
  (ontable shot106)
  (ontable shot311)
  (ontable shot168)
  (dispenses dispenser422 ingredient287)
  (dispenses dispenser462 ingredient251)
  (dispenses dispenser249 ingredient88)
  (clean shaker358)
  (clean shot106)
  (clean shot311)
  (clean shot168)
  (empty shaker358)
  (empty shot106)
  (empty shot311)
  (empty shot168)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker358 l0)
  (shaker-level shaker358 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail258 ingredient251)
  (cocktail-part2 cocktail258 ingredient287)
)
 (:goal
  (and
      (contains shot106 cocktail258)
      (contains shot311 ingredient88)
)))
