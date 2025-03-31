(define (problem prob)
 (:domain barman)
 (:objects 
      shaker133 - shaker
      left right - hand
      shot209 shot204 - shot
      ingredient116 ingredient78 ingredient29 - ingredient
      cocktail1 - cocktail
      dispenser298 dispenser112 dispenser93 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker133)
  (ontable shot209)
  (ontable shot204)
  (dispenses dispenser298 ingredient116)
  (dispenses dispenser112 ingredient78)
  (dispenses dispenser93 ingredient29)
  (clean shaker133)
  (clean shot209)
  (clean shot204)
  (empty shaker133)
  (empty shot209)
  (empty shot204)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker133 l0)
  (shaker-level shaker133 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient116)
  (cocktail-part2 cocktail1 ingredient29)
)
 (:goal
  (and
      (contains shot209 cocktail1)
)))
