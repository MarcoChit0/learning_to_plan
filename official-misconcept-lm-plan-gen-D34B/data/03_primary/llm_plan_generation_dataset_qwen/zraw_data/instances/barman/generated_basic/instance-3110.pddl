(define (problem prob)
 (:domain barman)
 (:objects 
      shaker277 - shaker
      left right - hand
      shot107 shot75 - shot
      ingredient494 ingredient251 - ingredient
      cocktail416 - cocktail
      dispenser323 dispenser68 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker277)
  (ontable shot107)
  (ontable shot75)
  (dispenses dispenser323 ingredient494)
  (dispenses dispenser68 ingredient251)
  (clean shaker277)
  (clean shot107)
  (clean shot75)
  (empty shaker277)
  (empty shot107)
  (empty shot75)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker277 l0)
  (shaker-level shaker277 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail416 ingredient494)
  (cocktail-part2 cocktail416 ingredient251)
)
 (:goal
  (and
      (contains shot107 cocktail416)
)))
