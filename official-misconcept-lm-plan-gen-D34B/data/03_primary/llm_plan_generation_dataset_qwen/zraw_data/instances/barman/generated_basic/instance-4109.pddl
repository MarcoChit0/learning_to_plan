(define (problem prob)
 (:domain barman)
 (:objects 
      shaker202 - shaker
      left right - hand
      shot191 shot266 - shot
      ingredient244 ingredient221 - ingredient
      cocktail382 - cocktail
      dispenser22 dispenser355 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker202)
  (ontable shot191)
  (ontable shot266)
  (dispenses dispenser22 ingredient244)
  (dispenses dispenser355 ingredient221)
  (clean shaker202)
  (clean shot191)
  (clean shot266)
  (empty shaker202)
  (empty shot191)
  (empty shot266)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker202 l0)
  (shaker-level shaker202 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail382 ingredient244)
  (cocktail-part2 cocktail382 ingredient221)
)
 (:goal
  (and
      (contains shot191 cocktail382)
)))
