(define (problem prob)
 (:domain barman)
 (:objects 
      shaker40 - shaker
      left right - hand
      shot7 shot246 - shot
      ingredient134 ingredient478 - ingredient
      cocktail266 - cocktail
      dispenser132 dispenser59 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker40)
  (ontable shot7)
  (ontable shot246)
  (dispenses dispenser132 ingredient134)
  (dispenses dispenser59 ingredient478)
  (clean shaker40)
  (clean shot7)
  (clean shot246)
  (empty shaker40)
  (empty shot7)
  (empty shot246)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker40 l0)
  (shaker-level shaker40 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail266 ingredient134)
  (cocktail-part2 cocktail266 ingredient478)
)
 (:goal
  (and
      (contains shot7 cocktail266)
)))
