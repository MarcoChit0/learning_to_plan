(define (problem prob)
 (:domain barman)
 (:objects 
      shaker88 - shaker
      left right - hand
      shot15 shot201 shot114 - shot
      ingredient100 ingredient232 ingredient62 ingredient241 - ingredient
      cocktail1 - cocktail
      dispenser199 dispenser425 dispenser407 dispenser3 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker88)
  (ontable shot15)
  (ontable shot201)
  (ontable shot114)
  (dispenses dispenser199 ingredient100)
  (dispenses dispenser425 ingredient232)
  (dispenses dispenser407 ingredient62)
  (dispenses dispenser3 ingredient241)
  (clean shaker88)
  (clean shot15)
  (clean shot201)
  (clean shot114)
  (empty shaker88)
  (empty shot15)
  (empty shot201)
  (empty shot114)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker88 l0)
  (shaker-level shaker88 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient232)
  (cocktail-part2 cocktail1 ingredient100)
)
 (:goal
  (and
      (contains shot15 cocktail1)
      (contains shot201 cocktail1)
)))
