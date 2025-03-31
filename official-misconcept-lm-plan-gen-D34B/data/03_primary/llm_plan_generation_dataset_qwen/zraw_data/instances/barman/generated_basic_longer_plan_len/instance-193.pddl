(define (problem prob)
 (:domain barman)
 (:objects 
      shaker450 - shaker
      left right - hand
      shot65 shot477 shot400 - shot
      ingredient488 ingredient202 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser47 dispenser134 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker450)
  (ontable shot65)
  (ontable shot477)
  (ontable shot400)
  (dispenses dispenser47 ingredient488)
  (dispenses dispenser134 ingredient202)
  (clean shaker450)
  (clean shot65)
  (clean shot477)
  (clean shot400)
  (empty shaker450)
  (empty shot65)
  (empty shot477)
  (empty shot400)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker450 l0)
  (shaker-level shaker450 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient488)
  (cocktail-part2 cocktail1 ingredient202)
  (cocktail-part1 cocktail2 ingredient202)
  (cocktail-part2 cocktail2 ingredient488)
)
 (:goal
  (and
      (contains shot65 cocktail1)
      (contains shot477 cocktail2)
)))
