(define (problem prob)
 (:domain barman)
 (:objects 
      shaker67 - shaker
      left right - hand
      shot200 shot18 - shot
      ingredient24 ingredient216 ingredient223 - ingredient
      cocktail306 - cocktail
      dispenser315 dispenser385 dispenser375 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker67)
  (ontable shot200)
  (ontable shot18)
  (dispenses dispenser315 ingredient24)
  (dispenses dispenser385 ingredient216)
  (dispenses dispenser375 ingredient223)
  (clean shaker67)
  (clean shot200)
  (clean shot18)
  (empty shaker67)
  (empty shot200)
  (empty shot18)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker67 l0)
  (shaker-level shaker67 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail306 ingredient24)
  (cocktail-part2 cocktail306 ingredient216)
)
 (:goal
  (and
      (contains shot200 cocktail306)
)))
