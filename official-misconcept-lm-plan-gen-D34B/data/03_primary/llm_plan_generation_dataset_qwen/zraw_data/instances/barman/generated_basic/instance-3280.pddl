(define (problem prob)
 (:domain barman)
 (:objects 
      shaker166 - shaker
      left right - hand
      shot421 shot202 - shot
      ingredient396 ingredient444 ingredient142 - ingredient
      cocktail374 - cocktail
      dispenser394 dispenser64 dispenser224 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker166)
  (ontable shot421)
  (ontable shot202)
  (dispenses dispenser394 ingredient396)
  (dispenses dispenser64 ingredient444)
  (dispenses dispenser224 ingredient142)
  (clean shaker166)
  (clean shot421)
  (clean shot202)
  (empty shaker166)
  (empty shot421)
  (empty shot202)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker166 l0)
  (shaker-level shaker166 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail374 ingredient396)
  (cocktail-part2 cocktail374 ingredient142)
)
 (:goal
  (and
      (contains shot421 cocktail374)
)))
