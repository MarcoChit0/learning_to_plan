(define (problem prob)
 (:domain barman)
 (:objects 
      shaker68 - shaker
      left right - hand
      shot118 - shot
      ingredient185 ingredient214 ingredient10 ingredient122 - ingredient
      cocktail149 - cocktail
      dispenser426 dispenser335 dispenser489 dispenser0 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker68)
  (ontable shot118)
  (dispenses dispenser426 ingredient185)
  (dispenses dispenser335 ingredient214)
  (dispenses dispenser489 ingredient10)
  (dispenses dispenser0 ingredient122)
  (clean shaker68)
  (clean shot118)
  (empty shaker68)
  (empty shot118)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker68 l0)
  (shaker-level shaker68 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail149 ingredient214)
  (cocktail-part2 cocktail149 ingredient185)
)
 (:goal
  (and
      (contains shot118 cocktail149)
)))
