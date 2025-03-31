(define (problem prob)
 (:domain barman)
 (:objects 
      shaker237 - shaker
      left right - hand
      shot359 - shot
      ingredient7 ingredient0 ingredient292 ingredient388 - ingredient
      cocktail25 - cocktail
      dispenser97 dispenser252 dispenser26 dispenser189 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker237)
  (ontable shot359)
  (dispenses dispenser97 ingredient7)
  (dispenses dispenser252 ingredient0)
  (dispenses dispenser26 ingredient292)
  (dispenses dispenser189 ingredient388)
  (clean shaker237)
  (clean shot359)
  (empty shaker237)
  (empty shot359)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker237 l0)
  (shaker-level shaker237 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail25 ingredient7)
  (cocktail-part2 cocktail25 ingredient292)
)
 (:goal
  (and
      (contains shot359 cocktail25)
)))
