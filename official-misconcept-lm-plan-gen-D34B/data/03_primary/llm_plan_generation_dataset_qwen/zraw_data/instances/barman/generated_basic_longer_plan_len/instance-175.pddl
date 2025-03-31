(define (problem prob)
 (:domain barman)
 (:objects 
      shaker360 - shaker
      left right - hand
      shot429 shot180 shot27 shot320 - shot
      ingredient62 ingredient16 - ingredient
      cocktail1 - cocktail
      dispenser253 dispenser314 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker360)
  (ontable shot429)
  (ontable shot180)
  (ontable shot27)
  (ontable shot320)
  (dispenses dispenser253 ingredient62)
  (dispenses dispenser314 ingredient16)
  (clean shaker360)
  (clean shot429)
  (clean shot180)
  (clean shot27)
  (clean shot320)
  (empty shaker360)
  (empty shot429)
  (empty shot180)
  (empty shot27)
  (empty shot320)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker360 l0)
  (shaker-level shaker360 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient16)
  (cocktail-part2 cocktail1 ingredient62)
)
 (:goal
  (and
      (contains shot429 cocktail1)
      (contains shot180 ingredient16)
      (contains shot27 ingredient16)
)))
