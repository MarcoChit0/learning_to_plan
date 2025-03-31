(define (problem prob)
 (:domain barman)
 (:objects 
      shaker90 - shaker
      left right - hand
      shot224 - shot
      ingredient50 ingredient317 - ingredient
      cocktail1 - cocktail
      dispenser416 dispenser246 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker90)
  (ontable shot224)
  (dispenses dispenser416 ingredient50)
  (dispenses dispenser246 ingredient317)
  (clean shaker90)
  (clean shot224)
  (empty shaker90)
  (empty shot224)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker90 l0)
  (shaker-level shaker90 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient50)
  (cocktail-part2 cocktail1 ingredient317)
)
 (:goal
  (and
      (contains shot224 cocktail1)
)))
