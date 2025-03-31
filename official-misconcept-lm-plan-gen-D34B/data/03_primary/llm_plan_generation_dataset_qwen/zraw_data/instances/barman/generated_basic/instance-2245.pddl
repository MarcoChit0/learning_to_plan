(define (problem prob)
 (:domain barman)
 (:objects 
      shaker68 - shaker
      left right - hand
      shot329 shot190 shot456 - shot
      ingredient177 ingredient441 ingredient226 ingredient254 - ingredient
      cocktail86 - cocktail
      dispenser92 dispenser304 dispenser243 dispenser208 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker68)
  (ontable shot329)
  (ontable shot190)
  (ontable shot456)
  (dispenses dispenser92 ingredient177)
  (dispenses dispenser304 ingredient441)
  (dispenses dispenser243 ingredient226)
  (dispenses dispenser208 ingredient254)
  (clean shaker68)
  (clean shot329)
  (clean shot190)
  (clean shot456)
  (empty shaker68)
  (empty shot329)
  (empty shot190)
  (empty shot456)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker68 l0)
  (shaker-level shaker68 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail86 ingredient441)
  (cocktail-part2 cocktail86 ingredient254)
)
 (:goal
  (and
      (contains shot329 cocktail86)
      (contains shot190 ingredient226)
)))
