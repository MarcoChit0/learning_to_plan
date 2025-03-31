(define (problem prob)
 (:domain barman)
 (:objects 
      shaker346 - shaker
      left right - hand
      shot363 shot440 shot188 - shot
      ingredient311 ingredient123 ingredient69 ingredient88 - ingredient
      cocktail478 - cocktail
      dispenser375 dispenser15 dispenser412 dispenser306 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker346)
  (ontable shot363)
  (ontable shot440)
  (ontable shot188)
  (dispenses dispenser375 ingredient311)
  (dispenses dispenser15 ingredient123)
  (dispenses dispenser412 ingredient69)
  (dispenses dispenser306 ingredient88)
  (clean shaker346)
  (clean shot363)
  (clean shot440)
  (clean shot188)
  (empty shaker346)
  (empty shot363)
  (empty shot440)
  (empty shot188)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker346 l0)
  (shaker-level shaker346 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail478 ingredient88)
  (cocktail-part2 cocktail478 ingredient69)
)
 (:goal
  (and
      (contains shot363 cocktail478)
      (contains shot440 cocktail478)
)))
