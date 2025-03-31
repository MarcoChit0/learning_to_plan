(define (problem prob)
 (:domain barman)
 (:objects 
      shaker42 - shaker
      left right - hand
      shot182 shot340 - shot
      ingredient435 ingredient9 ingredient128 - ingredient
      cocktail132 - cocktail
      dispenser365 dispenser161 dispenser356 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker42)
  (ontable shot182)
  (ontable shot340)
  (dispenses dispenser365 ingredient435)
  (dispenses dispenser161 ingredient9)
  (dispenses dispenser356 ingredient128)
  (clean shaker42)
  (clean shot182)
  (clean shot340)
  (empty shaker42)
  (empty shot182)
  (empty shot340)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker42 l0)
  (shaker-level shaker42 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail132 ingredient128)
  (cocktail-part2 cocktail132 ingredient435)
)
 (:goal
  (and
      (contains shot182 cocktail132)
)))
