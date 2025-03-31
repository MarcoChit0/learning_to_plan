(define (problem prob)
 (:domain barman)
 (:objects 
      shaker314 - shaker
      left right - hand
      shot149 shot411 shot251 - shot
      ingredient112 ingredient236 ingredient19 ingredient466 - ingredient
      cocktail1 - cocktail
      dispenser472 dispenser71 dispenser286 dispenser142 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker314)
  (ontable shot149)
  (ontable shot411)
  (ontable shot251)
  (dispenses dispenser472 ingredient112)
  (dispenses dispenser71 ingredient236)
  (dispenses dispenser286 ingredient19)
  (dispenses dispenser142 ingredient466)
  (clean shaker314)
  (clean shot149)
  (clean shot411)
  (clean shot251)
  (empty shaker314)
  (empty shot149)
  (empty shot411)
  (empty shot251)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker314 l0)
  (shaker-level shaker314 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient112)
  (cocktail-part2 cocktail1 ingredient236)
)
 (:goal
  (and
      (contains shot149 cocktail1)
      (contains shot411 ingredient466)
)))
