(define (problem prob)
 (:domain barman)
 (:objects 
      shaker132 - shaker
      left right - hand
      shot190 shot22 - shot
      ingredient317 ingredient73 - ingredient
      cocktail280 - cocktail
      dispenser122 dispenser259 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker132)
  (ontable shot190)
  (ontable shot22)
  (dispenses dispenser122 ingredient317)
  (dispenses dispenser259 ingredient73)
  (clean shaker132)
  (clean shot190)
  (clean shot22)
  (empty shaker132)
  (empty shot190)
  (empty shot22)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker132 l0)
  (shaker-level shaker132 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail280 ingredient317)
  (cocktail-part2 cocktail280 ingredient73)
)
 (:goal
  (and
      (contains shot190 cocktail280)
)))
