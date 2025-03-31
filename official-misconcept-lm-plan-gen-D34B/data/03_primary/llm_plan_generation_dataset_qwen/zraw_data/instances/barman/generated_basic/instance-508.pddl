(define (problem prob)
 (:domain barman)
 (:objects 
      shaker151 - shaker
      left right - hand
      shot274 shot46 - shot
      ingredient380 ingredient130 ingredient256 ingredient59 - ingredient
      cocktail1 - cocktail
      dispenser50 dispenser60 dispenser455 dispenser138 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker151)
  (ontable shot274)
  (ontable shot46)
  (dispenses dispenser50 ingredient380)
  (dispenses dispenser60 ingredient130)
  (dispenses dispenser455 ingredient256)
  (dispenses dispenser138 ingredient59)
  (clean shaker151)
  (clean shot274)
  (clean shot46)
  (empty shaker151)
  (empty shot274)
  (empty shot46)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker151 l0)
  (shaker-level shaker151 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient256)
  (cocktail-part2 cocktail1 ingredient59)
)
 (:goal
  (and
      (contains shot274 cocktail1)
)))
