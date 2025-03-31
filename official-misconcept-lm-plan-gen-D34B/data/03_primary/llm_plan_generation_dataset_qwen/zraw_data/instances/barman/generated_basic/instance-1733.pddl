(define (problem prob)
 (:domain barman)
 (:objects 
      shaker235 - shaker
      left right - hand
      shot339 shot112 - shot
      ingredient438 ingredient327 ingredient294 ingredient167 - ingredient
      cocktail260 - cocktail
      dispenser95 dispenser456 dispenser142 dispenser149 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker235)
  (ontable shot339)
  (ontable shot112)
  (dispenses dispenser95 ingredient438)
  (dispenses dispenser456 ingredient327)
  (dispenses dispenser142 ingredient294)
  (dispenses dispenser149 ingredient167)
  (clean shaker235)
  (clean shot339)
  (clean shot112)
  (empty shaker235)
  (empty shot339)
  (empty shot112)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker235 l0)
  (shaker-level shaker235 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail260 ingredient167)
  (cocktail-part2 cocktail260 ingredient294)
)
 (:goal
  (and
      (contains shot339 cocktail260)
)))
