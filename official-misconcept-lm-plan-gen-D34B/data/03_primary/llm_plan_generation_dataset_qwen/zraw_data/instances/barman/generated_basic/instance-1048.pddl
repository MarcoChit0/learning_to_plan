(define (problem prob)
 (:domain barman)
 (:objects 
      shaker477 - shaker
      left right - hand
      shot197 shot456 shot34 - shot
      ingredient497 ingredient230 ingredient254 ingredient75 - ingredient
      cocktail1 - cocktail
      dispenser36 dispenser350 dispenser495 dispenser372 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker477)
  (ontable shot197)
  (ontable shot456)
  (ontable shot34)
  (dispenses dispenser36 ingredient497)
  (dispenses dispenser350 ingredient230)
  (dispenses dispenser495 ingredient254)
  (dispenses dispenser372 ingredient75)
  (clean shaker477)
  (clean shot197)
  (clean shot456)
  (clean shot34)
  (empty shaker477)
  (empty shot197)
  (empty shot456)
  (empty shot34)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker477 l0)
  (shaker-level shaker477 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient254)
  (cocktail-part2 cocktail1 ingredient497)
)
 (:goal
  (and
      (contains shot197 cocktail1)
      (contains shot456 cocktail1)
)))
