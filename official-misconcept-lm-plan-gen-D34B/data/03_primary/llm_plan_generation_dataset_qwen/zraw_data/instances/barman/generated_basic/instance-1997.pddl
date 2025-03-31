(define (problem prob)
 (:domain barman)
 (:objects 
      shaker237 - shaker
      left right - hand
      shot168 shot333 - shot
      ingredient111 ingredient142 ingredient353 - ingredient
      cocktail370 - cocktail
      dispenser266 dispenser83 dispenser370 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker237)
  (ontable shot168)
  (ontable shot333)
  (dispenses dispenser266 ingredient111)
  (dispenses dispenser83 ingredient142)
  (dispenses dispenser370 ingredient353)
  (clean shaker237)
  (clean shot168)
  (clean shot333)
  (empty shaker237)
  (empty shot168)
  (empty shot333)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker237 l0)
  (shaker-level shaker237 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail370 ingredient142)
  (cocktail-part2 cocktail370 ingredient353)
)
 (:goal
  (and
      (contains shot168 cocktail370)
)))
