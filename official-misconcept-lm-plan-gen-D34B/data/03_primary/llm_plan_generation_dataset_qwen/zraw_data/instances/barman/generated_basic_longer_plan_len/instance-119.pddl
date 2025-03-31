(define (problem prob)
 (:domain barman)
 (:objects 
      shaker331 - shaker
      left right - hand
      shot419 shot205 shot184 - shot
      ingredient460 ingredient191 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser434 dispenser486 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker331)
  (ontable shot419)
  (ontable shot205)
  (ontable shot184)
  (dispenses dispenser434 ingredient460)
  (dispenses dispenser486 ingredient191)
  (clean shaker331)
  (clean shot419)
  (clean shot205)
  (clean shot184)
  (empty shaker331)
  (empty shot419)
  (empty shot205)
  (empty shot184)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker331 l0)
  (shaker-level shaker331 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient191)
  (cocktail-part2 cocktail1 ingredient460)
  (cocktail-part1 cocktail2 ingredient460)
  (cocktail-part2 cocktail2 ingredient191)
)
 (:goal
  (and
      (contains shot419 cocktail2)
      (contains shot205 cocktail1)
)))
