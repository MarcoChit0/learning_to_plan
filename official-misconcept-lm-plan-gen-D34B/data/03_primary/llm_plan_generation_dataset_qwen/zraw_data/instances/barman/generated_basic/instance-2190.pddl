(define (problem prob)
 (:domain barman)
 (:objects 
      shaker39 - shaker
      left right - hand
      shot480 shot350 shot477 - shot
      ingredient287 ingredient221 - ingredient
      cocktail45 - cocktail
      dispenser7 dispenser11 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker39)
  (ontable shot480)
  (ontable shot350)
  (ontable shot477)
  (dispenses dispenser7 ingredient287)
  (dispenses dispenser11 ingredient221)
  (clean shaker39)
  (clean shot480)
  (clean shot350)
  (clean shot477)
  (empty shaker39)
  (empty shot480)
  (empty shot350)
  (empty shot477)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker39 l0)
  (shaker-level shaker39 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail45 ingredient221)
  (cocktail-part2 cocktail45 ingredient287)
)
 (:goal
  (and
      (contains shot480 cocktail45)
      (contains shot350 ingredient287)
)))
