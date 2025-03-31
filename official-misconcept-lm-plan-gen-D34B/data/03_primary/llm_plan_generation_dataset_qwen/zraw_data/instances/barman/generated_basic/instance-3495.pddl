(define (problem prob)
 (:domain barman)
 (:objects 
      shaker30 - shaker
      left right - hand
      shot439 shot29 - shot
      ingredient90 ingredient221 ingredient275 ingredient80 - ingredient
      cocktail434 - cocktail
      dispenser311 dispenser229 dispenser465 dispenser156 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker30)
  (ontable shot439)
  (ontable shot29)
  (dispenses dispenser311 ingredient90)
  (dispenses dispenser229 ingredient221)
  (dispenses dispenser465 ingredient275)
  (dispenses dispenser156 ingredient80)
  (clean shaker30)
  (clean shot439)
  (clean shot29)
  (empty shaker30)
  (empty shot439)
  (empty shot29)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker30 l0)
  (shaker-level shaker30 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail434 ingredient90)
  (cocktail-part2 cocktail434 ingredient80)
)
 (:goal
  (and
      (contains shot439 cocktail434)
)))
