(define (problem prob)
 (:domain barman)
 (:objects 
      shaker346 - shaker
      left right - hand
      shot138 shot106 shot349 - shot
      ingredient173 ingredient38 - ingredient
      cocktail403 - cocktail
      dispenser251 dispenser367 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker346)
  (ontable shot138)
  (ontable shot106)
  (ontable shot349)
  (dispenses dispenser251 ingredient173)
  (dispenses dispenser367 ingredient38)
  (clean shaker346)
  (clean shot138)
  (clean shot106)
  (clean shot349)
  (empty shaker346)
  (empty shot138)
  (empty shot106)
  (empty shot349)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker346 l0)
  (shaker-level shaker346 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail403 ingredient38)
  (cocktail-part2 cocktail403 ingredient173)
)
 (:goal
  (and
      (contains shot138 cocktail403)
      (contains shot106 cocktail403)
)))
