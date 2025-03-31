(define (problem prob)
 (:domain barman)
 (:objects 
      shaker211 - shaker
      left right - hand
      shot196 shot162 shot434 - shot
      ingredient189 ingredient496 ingredient155 ingredient107 - ingredient
      cocktail1 - cocktail
      dispenser303 dispenser14 dispenser90 dispenser361 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker211)
  (ontable shot196)
  (ontable shot162)
  (ontable shot434)
  (dispenses dispenser303 ingredient189)
  (dispenses dispenser14 ingredient496)
  (dispenses dispenser90 ingredient155)
  (dispenses dispenser361 ingredient107)
  (clean shaker211)
  (clean shot196)
  (clean shot162)
  (clean shot434)
  (empty shaker211)
  (empty shot196)
  (empty shot162)
  (empty shot434)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker211 l0)
  (shaker-level shaker211 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient189)
  (cocktail-part2 cocktail1 ingredient155)
)
 (:goal
  (and
      (contains shot196 cocktail1)
      (contains shot162 cocktail1)
)))
