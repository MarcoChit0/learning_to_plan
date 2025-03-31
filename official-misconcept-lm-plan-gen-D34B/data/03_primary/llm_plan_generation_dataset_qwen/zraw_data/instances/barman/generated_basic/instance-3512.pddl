(define (problem prob)
 (:domain barman)
 (:objects 
      shaker255 - shaker
      left right - hand
      shot205 shot278 - shot
      ingredient42 ingredient245 ingredient24 ingredient147 - ingredient
      cocktail391 - cocktail
      dispenser36 dispenser239 dispenser60 dispenser35 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker255)
  (ontable shot205)
  (ontable shot278)
  (dispenses dispenser36 ingredient42)
  (dispenses dispenser239 ingredient245)
  (dispenses dispenser60 ingredient24)
  (dispenses dispenser35 ingredient147)
  (clean shaker255)
  (clean shot205)
  (clean shot278)
  (empty shaker255)
  (empty shot205)
  (empty shot278)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker255 l0)
  (shaker-level shaker255 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail391 ingredient147)
  (cocktail-part2 cocktail391 ingredient24)
)
 (:goal
  (and
      (contains shot205 cocktail391)
)))
