(define (problem prob)
 (:domain barman)
 (:objects 
      shaker115 - shaker
      left right - hand
      shot327 shot122 - shot
      ingredient443 ingredient458 ingredient92 - ingredient
      cocktail439 - cocktail
      dispenser405 dispenser60 dispenser432 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker115)
  (ontable shot327)
  (ontable shot122)
  (dispenses dispenser405 ingredient443)
  (dispenses dispenser60 ingredient458)
  (dispenses dispenser432 ingredient92)
  (clean shaker115)
  (clean shot327)
  (clean shot122)
  (empty shaker115)
  (empty shot327)
  (empty shot122)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker115 l0)
  (shaker-level shaker115 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail439 ingredient443)
  (cocktail-part2 cocktail439 ingredient92)
)
 (:goal
  (and
      (contains shot327 cocktail439)
)))
