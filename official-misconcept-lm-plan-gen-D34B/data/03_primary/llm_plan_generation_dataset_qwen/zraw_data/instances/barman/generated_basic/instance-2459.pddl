(define (problem prob)
 (:domain barman)
 (:objects 
      shaker209 - shaker
      left right - hand
      shot260 shot374 shot148 - shot
      ingredient478 ingredient425 ingredient171 ingredient406 - ingredient
      cocktail53 - cocktail
      dispenser151 dispenser379 dispenser316 dispenser97 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker209)
  (ontable shot260)
  (ontable shot374)
  (ontable shot148)
  (dispenses dispenser151 ingredient478)
  (dispenses dispenser379 ingredient425)
  (dispenses dispenser316 ingredient171)
  (dispenses dispenser97 ingredient406)
  (clean shaker209)
  (clean shot260)
  (clean shot374)
  (clean shot148)
  (empty shaker209)
  (empty shot260)
  (empty shot374)
  (empty shot148)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker209 l0)
  (shaker-level shaker209 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail53 ingredient478)
  (cocktail-part2 cocktail53 ingredient171)
)
 (:goal
  (and
      (contains shot260 cocktail53)
      (contains shot374 cocktail53)
)))
