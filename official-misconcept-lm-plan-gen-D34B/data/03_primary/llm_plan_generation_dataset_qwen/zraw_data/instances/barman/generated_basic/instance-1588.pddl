(define (problem prob)
 (:domain barman)
 (:objects 
      shaker331 - shaker
      left right - hand
      shot416 shot346 - shot
      ingredient347 ingredient264 - ingredient
      cocktail271 - cocktail
      dispenser236 dispenser76 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker331)
  (ontable shot416)
  (ontable shot346)
  (dispenses dispenser236 ingredient347)
  (dispenses dispenser76 ingredient264)
  (clean shaker331)
  (clean shot416)
  (clean shot346)
  (empty shaker331)
  (empty shot416)
  (empty shot346)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker331 l0)
  (shaker-level shaker331 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail271 ingredient347)
  (cocktail-part2 cocktail271 ingredient264)
)
 (:goal
  (and
      (contains shot416 cocktail271)
)))
