(define (problem prob)
 (:domain barman)
 (:objects 
      shaker450 - shaker
      left right - hand
      shot253 shot53 shot110 - shot
      ingredient157 ingredient221 ingredient447 ingredient68 - ingredient
      cocktail1 - cocktail
      dispenser441 dispenser494 dispenser362 dispenser196 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker450)
  (ontable shot253)
  (ontable shot53)
  (ontable shot110)
  (dispenses dispenser441 ingredient157)
  (dispenses dispenser494 ingredient221)
  (dispenses dispenser362 ingredient447)
  (dispenses dispenser196 ingredient68)
  (clean shaker450)
  (clean shot253)
  (clean shot53)
  (clean shot110)
  (empty shaker450)
  (empty shot253)
  (empty shot53)
  (empty shot110)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker450 l0)
  (shaker-level shaker450 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient157)
  (cocktail-part2 cocktail1 ingredient221)
)
 (:goal
  (and
      (contains shot253 cocktail1)
      (contains shot53 ingredient447)
)))
