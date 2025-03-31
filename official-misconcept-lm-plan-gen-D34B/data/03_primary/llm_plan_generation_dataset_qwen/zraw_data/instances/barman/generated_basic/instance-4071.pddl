(define (problem prob)
 (:domain barman)
 (:objects 
      shaker411 - shaker
      left right - hand
      shot190 shot104 - shot
      ingredient23 ingredient229 - ingredient
      cocktail399 - cocktail
      dispenser467 dispenser73 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker411)
  (ontable shot190)
  (ontable shot104)
  (dispenses dispenser467 ingredient23)
  (dispenses dispenser73 ingredient229)
  (clean shaker411)
  (clean shot190)
  (clean shot104)
  (empty shaker411)
  (empty shot190)
  (empty shot104)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker411 l0)
  (shaker-level shaker411 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail399 ingredient23)
  (cocktail-part2 cocktail399 ingredient229)
)
 (:goal
  (and
      (contains shot190 cocktail399)
)))
