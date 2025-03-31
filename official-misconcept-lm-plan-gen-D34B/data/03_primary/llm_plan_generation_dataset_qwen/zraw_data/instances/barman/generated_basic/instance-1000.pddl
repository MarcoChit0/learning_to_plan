(define (problem prob)
 (:domain barman)
 (:objects 
      shaker173 - shaker
      left right - hand
      shot144 shot189 - shot
      ingredient41 ingredient200 - ingredient
      cocktail1 - cocktail
      dispenser470 dispenser86 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker173)
  (ontable shot144)
  (ontable shot189)
  (dispenses dispenser470 ingredient41)
  (dispenses dispenser86 ingredient200)
  (clean shaker173)
  (clean shot144)
  (clean shot189)
  (empty shaker173)
  (empty shot144)
  (empty shot189)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker173 l0)
  (shaker-level shaker173 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient200)
  (cocktail-part2 cocktail1 ingredient41)
)
 (:goal
  (and
      (contains shot144 cocktail1)
)))
