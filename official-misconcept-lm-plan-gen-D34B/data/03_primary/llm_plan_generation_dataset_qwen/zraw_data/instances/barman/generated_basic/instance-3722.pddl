(define (problem prob)
 (:domain barman)
 (:objects 
      shaker151 - shaker
      left right - hand
      shot490 - shot
      ingredient214 ingredient295 ingredient58 ingredient423 - ingredient
      cocktail399 - cocktail
      dispenser339 dispenser234 dispenser424 dispenser418 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker151)
  (ontable shot490)
  (dispenses dispenser339 ingredient214)
  (dispenses dispenser234 ingredient295)
  (dispenses dispenser424 ingredient58)
  (dispenses dispenser418 ingredient423)
  (clean shaker151)
  (clean shot490)
  (empty shaker151)
  (empty shot490)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker151 l0)
  (shaker-level shaker151 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail399 ingredient214)
  (cocktail-part2 cocktail399 ingredient295)
)
 (:goal
  (and
      (contains shot490 cocktail399)
)))
