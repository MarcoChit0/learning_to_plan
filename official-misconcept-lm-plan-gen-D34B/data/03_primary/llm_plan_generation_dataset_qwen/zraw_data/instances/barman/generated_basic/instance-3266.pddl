(define (problem prob)
 (:domain barman)
 (:objects 
      shaker202 - shaker
      left right - hand
      shot170 shot8 shot307 - shot
      ingredient407 ingredient226 ingredient24 ingredient267 - ingredient
      cocktail247 - cocktail
      dispenser252 dispenser394 dispenser179 dispenser203 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker202)
  (ontable shot170)
  (ontable shot8)
  (ontable shot307)
  (dispenses dispenser252 ingredient407)
  (dispenses dispenser394 ingredient226)
  (dispenses dispenser179 ingredient24)
  (dispenses dispenser203 ingredient267)
  (clean shaker202)
  (clean shot170)
  (clean shot8)
  (clean shot307)
  (empty shaker202)
  (empty shot170)
  (empty shot8)
  (empty shot307)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker202 l0)
  (shaker-level shaker202 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail247 ingredient267)
  (cocktail-part2 cocktail247 ingredient24)
)
 (:goal
  (and
      (contains shot170 cocktail247)
      (contains shot8 ingredient267)
)))
