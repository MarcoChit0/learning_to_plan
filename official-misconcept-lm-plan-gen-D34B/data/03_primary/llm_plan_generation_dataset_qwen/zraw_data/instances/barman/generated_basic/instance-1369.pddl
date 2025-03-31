(define (problem prob)
 (:domain barman)
 (:objects 
      shaker290 - shaker
      left right - hand
      shot214 - shot
      ingredient269 ingredient317 ingredient102 - ingredient
      cocktail241 - cocktail
      dispenser252 dispenser142 dispenser463 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker290)
  (ontable shot214)
  (dispenses dispenser252 ingredient269)
  (dispenses dispenser142 ingredient317)
  (dispenses dispenser463 ingredient102)
  (clean shaker290)
  (clean shot214)
  (empty shaker290)
  (empty shot214)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker290 l0)
  (shaker-level shaker290 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail241 ingredient269)
  (cocktail-part2 cocktail241 ingredient317)
)
 (:goal
  (and
      (contains shot214 cocktail241)
)))
