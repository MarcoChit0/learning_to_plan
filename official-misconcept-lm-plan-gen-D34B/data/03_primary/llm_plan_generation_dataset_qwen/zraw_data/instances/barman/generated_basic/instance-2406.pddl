(define (problem prob)
 (:domain barman)
 (:objects 
      shaker317 - shaker
      left right - hand
      shot421 - shot
      ingredient45 ingredient192 - ingredient
      cocktail212 - cocktail
      dispenser182 dispenser369 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker317)
  (ontable shot421)
  (dispenses dispenser182 ingredient45)
  (dispenses dispenser369 ingredient192)
  (clean shaker317)
  (clean shot421)
  (empty shaker317)
  (empty shot421)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker317 l0)
  (shaker-level shaker317 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail212 ingredient192)
  (cocktail-part2 cocktail212 ingredient45)
)
 (:goal
  (and
      (contains shot421 cocktail212)
)))
