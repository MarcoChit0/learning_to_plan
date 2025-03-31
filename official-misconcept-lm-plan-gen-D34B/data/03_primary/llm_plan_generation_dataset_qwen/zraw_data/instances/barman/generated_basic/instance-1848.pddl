(define (problem prob)
 (:domain barman)
 (:objects 
      shaker424 - shaker
      left right - hand
      shot392 shot482 - shot
      ingredient371 ingredient30 ingredient443 - ingredient
      cocktail154 - cocktail
      dispenser81 dispenser46 dispenser468 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker424)
  (ontable shot392)
  (ontable shot482)
  (dispenses dispenser81 ingredient371)
  (dispenses dispenser46 ingredient30)
  (dispenses dispenser468 ingredient443)
  (clean shaker424)
  (clean shot392)
  (clean shot482)
  (empty shaker424)
  (empty shot392)
  (empty shot482)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker424 l0)
  (shaker-level shaker424 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail154 ingredient443)
  (cocktail-part2 cocktail154 ingredient371)
)
 (:goal
  (and
      (contains shot392 cocktail154)
)))
