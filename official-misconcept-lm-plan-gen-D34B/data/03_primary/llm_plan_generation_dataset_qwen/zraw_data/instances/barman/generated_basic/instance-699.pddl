(define (problem prob)
 (:domain barman)
 (:objects 
      shaker45 - shaker
      left right - hand
      shot160 shot172 - shot
      ingredient443 ingredient5 ingredient396 - ingredient
      cocktail1 - cocktail
      dispenser73 dispenser347 dispenser341 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker45)
  (ontable shot160)
  (ontable shot172)
  (dispenses dispenser73 ingredient443)
  (dispenses dispenser347 ingredient5)
  (dispenses dispenser341 ingredient396)
  (clean shaker45)
  (clean shot160)
  (clean shot172)
  (empty shaker45)
  (empty shot160)
  (empty shot172)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker45 l0)
  (shaker-level shaker45 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient396)
  (cocktail-part2 cocktail1 ingredient443)
)
 (:goal
  (and
      (contains shot160 cocktail1)
)))
