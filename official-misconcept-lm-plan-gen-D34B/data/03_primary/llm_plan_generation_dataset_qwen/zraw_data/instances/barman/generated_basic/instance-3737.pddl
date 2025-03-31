(define (problem prob)
 (:domain barman)
 (:objects 
      shaker18 - shaker
      left right - hand
      shot201 shot417 shot12 - shot
      ingredient356 ingredient86 ingredient438 - ingredient
      cocktail375 - cocktail
      dispenser256 dispenser265 dispenser207 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker18)
  (ontable shot201)
  (ontable shot417)
  (ontable shot12)
  (dispenses dispenser256 ingredient356)
  (dispenses dispenser265 ingredient86)
  (dispenses dispenser207 ingredient438)
  (clean shaker18)
  (clean shot201)
  (clean shot417)
  (clean shot12)
  (empty shaker18)
  (empty shot201)
  (empty shot417)
  (empty shot12)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker18 l0)
  (shaker-level shaker18 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail375 ingredient438)
  (cocktail-part2 cocktail375 ingredient356)
)
 (:goal
  (and
      (contains shot201 cocktail375)
      (contains shot417 ingredient86)
)))
