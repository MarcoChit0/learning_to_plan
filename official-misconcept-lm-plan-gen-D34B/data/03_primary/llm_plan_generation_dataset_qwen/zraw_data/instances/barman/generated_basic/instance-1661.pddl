(define (problem prob)
 (:domain barman)
 (:objects 
      shaker329 - shaker
      left right - hand
      shot404 shot312 - shot
      ingredient251 ingredient427 ingredient0 ingredient472 - ingredient
      cocktail275 - cocktail
      dispenser160 dispenser127 dispenser207 dispenser226 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker329)
  (ontable shot404)
  (ontable shot312)
  (dispenses dispenser160 ingredient251)
  (dispenses dispenser127 ingredient427)
  (dispenses dispenser207 ingredient0)
  (dispenses dispenser226 ingredient472)
  (clean shaker329)
  (clean shot404)
  (clean shot312)
  (empty shaker329)
  (empty shot404)
  (empty shot312)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker329 l0)
  (shaker-level shaker329 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail275 ingredient472)
  (cocktail-part2 cocktail275 ingredient251)
)
 (:goal
  (and
      (contains shot404 cocktail275)
)))
