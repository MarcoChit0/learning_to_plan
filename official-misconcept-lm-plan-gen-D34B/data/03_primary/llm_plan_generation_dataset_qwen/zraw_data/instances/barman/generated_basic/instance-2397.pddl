(define (problem prob)
 (:domain barman)
 (:objects 
      shaker209 - shaker
      left right - hand
      shot159 shot404 - shot
      ingredient313 ingredient156 ingredient247 - ingredient
      cocktail225 - cocktail
      dispenser439 dispenser97 dispenser490 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker209)
  (ontable shot159)
  (ontable shot404)
  (dispenses dispenser439 ingredient313)
  (dispenses dispenser97 ingredient156)
  (dispenses dispenser490 ingredient247)
  (clean shaker209)
  (clean shot159)
  (clean shot404)
  (empty shaker209)
  (empty shot159)
  (empty shot404)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker209 l0)
  (shaker-level shaker209 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail225 ingredient156)
  (cocktail-part2 cocktail225 ingredient313)
)
 (:goal
  (and
      (contains shot159 cocktail225)
)))
