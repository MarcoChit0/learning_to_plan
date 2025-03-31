(define (problem prob)
 (:domain barman)
 (:objects 
      shaker260 - shaker
      left right - hand
      shot189 shot260 - shot
      ingredient229 ingredient318 - ingredient
      cocktail1 - cocktail
      dispenser197 dispenser274 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker260)
  (ontable shot189)
  (ontable shot260)
  (dispenses dispenser197 ingredient229)
  (dispenses dispenser274 ingredient318)
  (clean shaker260)
  (clean shot189)
  (clean shot260)
  (empty shaker260)
  (empty shot189)
  (empty shot260)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker260 l0)
  (shaker-level shaker260 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient229)
  (cocktail-part2 cocktail1 ingredient318)
)
 (:goal
  (and
      (contains shot189 cocktail1)
)))
