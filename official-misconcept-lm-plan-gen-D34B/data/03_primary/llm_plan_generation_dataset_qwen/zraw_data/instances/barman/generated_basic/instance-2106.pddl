(define (problem prob)
 (:domain barman)
 (:objects 
      shaker417 - shaker
      left right - hand
      shot255 shot295 shot119 - shot
      ingredient171 ingredient396 - ingredient
      cocktail149 - cocktail
      dispenser309 dispenser95 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker417)
  (ontable shot255)
  (ontable shot295)
  (ontable shot119)
  (dispenses dispenser309 ingredient171)
  (dispenses dispenser95 ingredient396)
  (clean shaker417)
  (clean shot255)
  (clean shot295)
  (clean shot119)
  (empty shaker417)
  (empty shot255)
  (empty shot295)
  (empty shot119)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker417 l0)
  (shaker-level shaker417 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail149 ingredient171)
  (cocktail-part2 cocktail149 ingredient396)
)
 (:goal
  (and
      (contains shot255 cocktail149)
      (contains shot295 cocktail149)
)))
