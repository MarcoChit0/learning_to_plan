(define (problem prob)
 (:domain barman)
 (:objects 
      shaker121 - shaker
      left right - hand
      shot330 shot81 shot226 - shot
      ingredient78 ingredient294 - ingredient
      cocktail235 - cocktail
      dispenser48 dispenser32 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker121)
  (ontable shot330)
  (ontable shot81)
  (ontable shot226)
  (dispenses dispenser48 ingredient78)
  (dispenses dispenser32 ingredient294)
  (clean shaker121)
  (clean shot330)
  (clean shot81)
  (clean shot226)
  (empty shaker121)
  (empty shot330)
  (empty shot81)
  (empty shot226)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker121 l0)
  (shaker-level shaker121 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail235 ingredient78)
  (cocktail-part2 cocktail235 ingredient294)
)
 (:goal
  (and
      (contains shot330 cocktail235)
      (contains shot81 cocktail235)
)))
