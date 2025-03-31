(define (problem prob)
 (:domain barman)
 (:objects 
      shaker132 - shaker
      left right - hand
      shot163 shot319 - shot
      ingredient115 ingredient181 ingredient425 ingredient428 - ingredient
      cocktail475 - cocktail
      dispenser57 dispenser371 dispenser270 dispenser89 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker132)
  (ontable shot163)
  (ontable shot319)
  (dispenses dispenser57 ingredient115)
  (dispenses dispenser371 ingredient181)
  (dispenses dispenser270 ingredient425)
  (dispenses dispenser89 ingredient428)
  (clean shaker132)
  (clean shot163)
  (clean shot319)
  (empty shaker132)
  (empty shot163)
  (empty shot319)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker132 l0)
  (shaker-level shaker132 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail475 ingredient425)
  (cocktail-part2 cocktail475 ingredient181)
)
 (:goal
  (and
      (contains shot163 cocktail475)
)))
