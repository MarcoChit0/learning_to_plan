(define (problem prob)
 (:domain barman)
 (:objects 
      shaker158 - shaker
      left right - hand
      shot342 shot155 - shot
      ingredient244 ingredient473 - ingredient
      cocktail290 - cocktail
      dispenser183 dispenser344 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker158)
  (ontable shot342)
  (ontable shot155)
  (dispenses dispenser183 ingredient244)
  (dispenses dispenser344 ingredient473)
  (clean shaker158)
  (clean shot342)
  (clean shot155)
  (empty shaker158)
  (empty shot342)
  (empty shot155)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker158 l0)
  (shaker-level shaker158 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail290 ingredient473)
  (cocktail-part2 cocktail290 ingredient244)
)
 (:goal
  (and
      (contains shot342 cocktail290)
)))
