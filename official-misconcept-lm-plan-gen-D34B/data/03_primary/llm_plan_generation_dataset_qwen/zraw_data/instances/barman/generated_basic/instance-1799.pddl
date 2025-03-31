(define (problem prob)
 (:domain barman)
 (:objects 
      shaker219 - shaker
      left right - hand
      shot275 shot8 - shot
      ingredient445 ingredient487 ingredient377 ingredient499 - ingredient
      cocktail201 - cocktail
      dispenser204 dispenser427 dispenser18 dispenser436 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker219)
  (ontable shot275)
  (ontable shot8)
  (dispenses dispenser204 ingredient445)
  (dispenses dispenser427 ingredient487)
  (dispenses dispenser18 ingredient377)
  (dispenses dispenser436 ingredient499)
  (clean shaker219)
  (clean shot275)
  (clean shot8)
  (empty shaker219)
  (empty shot275)
  (empty shot8)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker219 l0)
  (shaker-level shaker219 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail201 ingredient499)
  (cocktail-part2 cocktail201 ingredient377)
)
 (:goal
  (and
      (contains shot275 cocktail201)
)))
