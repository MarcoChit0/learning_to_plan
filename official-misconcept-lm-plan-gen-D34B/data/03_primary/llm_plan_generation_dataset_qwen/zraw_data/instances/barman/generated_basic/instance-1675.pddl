(define (problem prob)
 (:domain barman)
 (:objects 
      shaker119 - shaker
      left right - hand
      shot217 shot295 - shot
      ingredient327 ingredient45 ingredient146 ingredient278 - ingredient
      cocktail308 - cocktail
      dispenser349 dispenser244 dispenser176 dispenser262 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker119)
  (ontable shot217)
  (ontable shot295)
  (dispenses dispenser349 ingredient327)
  (dispenses dispenser244 ingredient45)
  (dispenses dispenser176 ingredient146)
  (dispenses dispenser262 ingredient278)
  (clean shaker119)
  (clean shot217)
  (clean shot295)
  (empty shaker119)
  (empty shot217)
  (empty shot295)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker119 l0)
  (shaker-level shaker119 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail308 ingredient146)
  (cocktail-part2 cocktail308 ingredient45)
)
 (:goal
  (and
      (contains shot217 cocktail308)
)))
