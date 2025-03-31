(define (problem prob)
 (:domain barman)
 (:objects 
      shaker388 - shaker
      left right - hand
      shot401 shot240 - shot
      ingredient267 ingredient291 ingredient368 - ingredient
      cocktail8 - cocktail
      dispenser45 dispenser95 dispenser310 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker388)
  (ontable shot401)
  (ontable shot240)
  (dispenses dispenser45 ingredient267)
  (dispenses dispenser95 ingredient291)
  (dispenses dispenser310 ingredient368)
  (clean shaker388)
  (clean shot401)
  (clean shot240)
  (empty shaker388)
  (empty shot401)
  (empty shot240)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker388 l0)
  (shaker-level shaker388 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail8 ingredient368)
  (cocktail-part2 cocktail8 ingredient291)
)
 (:goal
  (and
      (contains shot401 cocktail8)
)))
