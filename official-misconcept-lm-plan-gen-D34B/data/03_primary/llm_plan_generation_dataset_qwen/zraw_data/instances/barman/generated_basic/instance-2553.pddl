(define (problem prob)
 (:domain barman)
 (:objects 
      shaker0 - shaker
      left right - hand
      shot231 shot428 shot82 - shot
      ingredient274 ingredient468 - ingredient
      cocktail388 - cocktail
      dispenser45 dispenser144 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker0)
  (ontable shot231)
  (ontable shot428)
  (ontable shot82)
  (dispenses dispenser45 ingredient274)
  (dispenses dispenser144 ingredient468)
  (clean shaker0)
  (clean shot231)
  (clean shot428)
  (clean shot82)
  (empty shaker0)
  (empty shot231)
  (empty shot428)
  (empty shot82)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker0 l0)
  (shaker-level shaker0 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail388 ingredient468)
  (cocktail-part2 cocktail388 ingredient274)
)
 (:goal
  (and
      (contains shot231 cocktail388)
      (contains shot428 cocktail388)
)))
